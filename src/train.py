import unsloth  # noqa: I001 F401

import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BatchEncoding, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer

import src.utils as utils
from src.models import Model


class Args(Tap):
    model_name: str = "rinna/japanese-gpt-neox-3.6b"
    dataset_dir: Path = "./datasets/livedoor"

    batch_size: int = 32
    epochs: int = 10
    num_warmup_epochs: int = 1

    use_unsloth: bool = False

    template_type: int = 2

    lr: float = 5e-4
    lora_r: int = 32
    weight_decay: float = 0.01
    max_seq_len: int = 512
    gradient_checkpointing: bool = True

    seed: int = 42

    def process_args(self):
        self.label2id: dict[str, int] = utils.load_json(
            self.dataset_dir / "label2id.json"
        )
        self.labels: list[int] = list(self.label2id.values())

        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
        self.output_dir = self._make_output_dir(
            "outputs",
            self.model_name,
            date,
            time,
        )

    def _make_output_dir(self, *args) -> Path:
        args = [str(a).replace("/", "__") for a in args]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True)
        return output_dir


class Experiment:
    def __init__(self, args: Args):
        self.args: Args = args

        use_fast = "japanese-gpt-neox" not in args.model_name
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=args.max_seq_len,
            use_fast=use_fast,
        )

        self.model: PreTrainedModel = Model(
            model_name=args.model_name,
            num_labels=len(args.labels),
            lora_r=args.lora_r,
            max_seq_len=args.max_seq_len,
            use_unsloth=args.use_unsloth,
            gradient_checkpointing=args.gradient_checkpointing,
        ).eval()
        self.model.write_trainable_params()

        self.train_dataloader = self.load_dataset(split="train", shuffle=True)
        steps_per_epoch: int = len(self.train_dataloader)

        self.accelerator = Accelerator(log_with="mlflow")
        (
            self.model,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
            self.load_dataset(split="val", shuffle=False),
            self.load_dataset(split="test", shuffle=False),
            *self.create_optimizer(steps_per_epoch),
        )

        mlflow.log_metrics(
            {
                "train.docs": len(self.train_dataloader.dataset),
                "valid.docs": len(self.valid_dataloader.dataset),
                "test.docs": len(self.test_dataloader.dataset),
            }
        )

    def load_dataset(
        self,
        split: str,
        shuffle: bool = False,
    ) -> DataLoader:
        path: Path = self.args.dataset_dir / f"{split}.jsonl"
        dataset: list[dict] = utils.load_jsonl(path).to_dict(orient="records")
        return self.create_loader(dataset, shuffle=shuffle)

    def build_input(self, title: str, body: str) -> str:
        if self.args.template_type == 0:
            return f"タイトル: {title}\n本文: {body}\nラベル: "
        elif self.args.template_type == 1:
            return f"タイトル: {title}\n本文: {body}"
        elif self.args.template_type == 2:
            return f"{title}\n{body}"

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        title = [d["title"] for d in data_list]
        body = [d["body"] for d in data_list]
        text = [self.build_input(t, b) for t, b in zip(title, body)]
        text_length = [len(t) for t in text]

        inputs: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

        labels = torch.LongTensor([d["label"] for d in data_list])
        text_length = torch.IntTensor([len(t) for t in text])
        return BatchEncoding({**inputs, "labels": labels, "text_length": text_length})

    def create_loader(
        self,
        dataset,
        batch_size=None,
        shuffle=False,
    ):
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size or self.args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    def create_optimizer(
        self,
        steps_per_epoch: int,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if name not in no_decay
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if name in no_decay
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=steps_per_epoch * self.args.num_warmup_epochs,
            num_training_steps=steps_per_epoch * self.args.epochs,
        )

        return optimizer, lr_scheduler

    def run(self):
        # setup log
        mlflow.set_experiment("train")
        mlflow.system_metrics.enable_system_metrics_logging()
        mlflow.start_run()
        mlflow.log_params(self.args.as_dict())

        metrics = {
            "epoch": -1,
            "train.loss": np.inf,
            **{f"valid.{k}": v for k, v in self.evaluate(self.val_dataloader).items()},
        }

        best_epoch, best_val_f1 = None, metrics["valid.f1"]
        self.log(metrics)

        for epoch in trange(self.args.epochs, dynamic_ncols=True):
            self.model.train()

            ts_start = time.perf_counter()
            total_loss, total_token, total_text_length = 0, 0, 0

            for batch in tqdm(
                self.train_dataloader,
                total=len(self.train_dataloader),
                dynamic_ncols=True,
                leave=False,
            ):
                text_length = batch["text_length"].tolist()
                total_text_length += sum(text_length)
                batch_tokens = batch["attention_mask"].sum(dim=1).tolist()
                total_token += sum(batch_tokens)

                for query in ["token_type_ids", "text_length"]:
                    if query in batch:
                        batch.pop(query)

                self.optimizer.zero_grad()
                out: SequenceClassifierOutput = self.model(**batch)
                loss: torch.FloatTensor = out.loss

                batch_size: int = batch.input_ids.size(0)
                total_loss += loss.item() * batch_size

                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()

            train_elapsed_time = time.perf_counter() - ts_start

            self.model.eval()

            dataset_length = len(self.train_dataloader.dataset)

            metrics = {
                "epoch": epoch,
                "train.loss": total_loss / dataset_length,
                "train.elapsed_time": train_elapsed_time,
                "train.tokens": total_token,
                "train.avg_tokens_par_sec": total_token / train_elapsed_time,
                "train.avg_data_par_sec": dataset_length / train_elapsed_time,
                "train.avg_tokens_par_doc": total_token / dataset_length,
                "train.text_length": total_text_length,
                "train.avg_text_length": total_text_length / dataset_length,
                "train.avg_text_length_par_token": total_text_length / total_token,
                **{
                    f"valid.{k}": v
                    for k, v in self.evaluate(self.val_dataloader).items()
                },
            }
            self.log(metrics)

            if metrics["valid.f1"] > best_val_f1:
                best_val_f1 = metrics["valid.f1"]
                best_epoch = epoch
                best_state_dict = self.model.clone_state_dict()

        self.model.load_state_dict(best_state_dict)
        self.model.eval()

        val_metrics = {
            "best.epoch": best_epoch,
            **{
                f"best.valid.{k}": v
                for k, v in self.evaluate(self.val_dataloader).items()
            },
        }
        test_metrics = {
            f"best.test.{k}": v for k, v in self.evaluate(self.test_dataloader).items()
        }

        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.end_run()

        return val_metrics, test_metrics

    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss, total_token, gold_labels, pred_labels = 0, 0, [], []
        total_text_length = 0

        ts_start = time.perf_counter()

        for batch in tqdm(
            dataloader, total=len(dataloader), dynamic_ncols=True, leave=False
        ):
            text_length = batch["text_length"].tolist()
            total_text_length += sum(text_length)
            batch_tokens = batch["attention_mask"].sum(dim=1).tolist()
            total_token += sum(batch_tokens)

            for query in ["token_type_ids", "text_length"]:
                if query in batch:
                    batch.pop(query)

            out: SequenceClassifierOutput = self.model(**batch)

            batch_size: int = batch.input_ids.size(0)
            loss = out.loss.item() * batch_size

            pred_labels += out.logits.argmax(dim=-1).tolist()
            gold_labels += batch.labels.tolist()
            total_loss += loss

        elapsed_time = time.perf_counter() - ts_start

        accuracy: float = accuracy_score(gold_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=args.labels,
        )
        dataset_length = len(dataloader.dataset)

        return {
            "loss": total_loss / dataset_length,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "elapsed_time": elapsed_time,
            "tokens": total_token,
            "avg_tokens_par_sec": total_token / elapsed_time,
            "avg_data_par_sec": dataset_length / elapsed_time,
            "avg_tokens_par_doc": total_token / dataset_length,
            "text_length": total_text_length,
            "avg_text_length": total_text_length / dataset_length,
            "avg_text_length_par_token": total_text_length / total_token,
        }

    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']}, "
            f"train.loss: {metrics['train.loss']:2.6f}, "
            f"valid.loss: {metrics['valid.loss']:2.6f}, "
            f"accuracy: {metrics['valid.accuracy']:.4f}, "
            f"precision: {metrics['valid.precision']:.4f}, "
            f"recall: {metrics['valid.recall']:.4f}, "
            f"f1: {metrics['valid.f1']:.4f}, "
        )
        mlflow.log_metrics(metrics, step=metrics["epoch"])


def main(args: Args):
    exp = Experiment(args=args)
    val_metrics, test_metrics = exp.run()

    utils.save_json(val_metrics, args.output_dir / "val-metrics.json")
    utils.save_json(test_metrics, args.output_dir / "test-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    args = Args().parse_args()
    utils.init(seed=args.seed)
    main(args)
