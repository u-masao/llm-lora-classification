# src/train.py

import time
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import torchinfo
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup

import src.utils as utils
from src.models import ClassificationModel


class Args(Tap):
    # --- Model & Tokenizer Arguments ---
    model_name: str = "rinna/japanese-gpt-neox-3.6b"
    max_seq_len: int = 512

    # --- LoRA Arguments ---
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # --- Data Arguments ---
    dataset_dir: Path = Path("./datasets/livedoor")
    template_type: int = 2

    # --- Training Arguments ---
    epochs: int = 10
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.01
    num_warmup_epochs: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True

    # --- Environment Arguments ---
    seed: int = 42
    num_workers: int = 4
    model_print_depth: int = 4

    def process_args(self):
        # ラベル情報をロード
        self.label2id: dict[str, int] = utils.load_json(
            self.dataset_dir / "label2id.json"
        )
        self.labels: list[int] = list(self.label2id.values())

        # 出力ディレクトリを生成
        date, time_str = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
        model_name_safe = self.model_name.replace("/", "__")
        self.output_dir = Path("outputs", model_name_safe, date, time_str)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class Trainer:
    def __init__(self, args: Args):
        self.args = args
        self.accelerator = Accelerator(
            log_with="mlflow",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        # --- 1. Tokenizer のロード ---
        use_fast = "japanese-gpt-neox" not in args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, use_fast=use_fast
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 2. Model の初期化 ---
        self.model = ClassificationModel(
            model_name=args.model_name,
            num_labels=len(args.labels),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        self._log_model_summary()

        # --- 3. Dataloader の準備 ---
        self.train_dataloader = self._create_dataloader(split="train", shuffle=True)
        self.val_dataloader = self._create_dataloader(split="val", shuffle=False)
        self.test_dataloader = self._create_dataloader(split="test", shuffle=False)

        # --- 4. Optimizer & Scheduler の準備 ---
        steps_per_epoch = len(self.train_dataloader) // args.gradient_accumulation_steps
        optimizer, lr_scheduler = self._create_optimizer(steps_per_epoch)

        # --- 5. Accelerator の準備 ---
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            optimizer,
            lr_scheduler,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        )
        self._log_initial_metrics()

    def _log_model_summary(self):
        """torchinfo を使ってモデルのサマリーをログに記録する"""
        if not self.accelerator.is_main_process:
            return

        # ダミー入力を作成してサマリーを生成
        dummy_input = {
            "input_ids": torch.zeros(1, self.args.max_seq_len, dtype=torch.long),
            "attention_mask": torch.zeros(1, self.args.max_seq_len, dtype=torch.long),
        }
        summary = torchinfo.summary(
            self.model,
            input_data=dummy_input,
            depth=self.args.model_print_depth,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=15,
            verbose=0,
        )

        tqdm.write(str(summary))
        mlflow.log_text(str(summary), "model_summary.txt")
        mlflow.log_metrics(
            {
                "model.total_params": summary.total_params,
                "model.trainable_params": summary.trainable_params,
            }
        )

    def _log_initial_metrics(self):
        """データセットサイズや Accelerator の設定をログに記録する"""
        if not self.accelerator.is_main_process:
            return

        mlflow.log_metrics(
            {
                "dataset.train_size": len(self.train_dataloader.dataset),
                "dataset.val_size": len(self.val_dataloader.dataset),
                "dataset.test_size": len(self.test_dataloader.dataset),
            }
        )
        mlflow.log_params(
            {
                "accelerate.distributed_type": str(self.accelerator.distributed_type),
                "accelerate.num_processes": self.accelerator.num_processes,
                "accelerate.mixed_precision": self.accelerator.mixed_precision,
            }
        )
        if self.accelerator.deepspeed_config:
            mlflow.log_dict(self.accelerator.deepspeed_config, "deepspeed_config.json")
        self.accelerator.wait_for_everyone()

    def _create_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = utils.load_jsonl(self.args.dataset_dir / f"{split}.jsonl").to_dict(
            "records"
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def _collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        """データからプロンプトを構築し、トークナイズしてバッチを作成する"""
        texts = [self._build_prompt(d["title"], d["body"]) for d in data_list]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_len,
            return_tensors="pt",
        )
        inputs["labels"] = torch.LongTensor([d["label"] for d in data_list])
        return inputs

    def _build_prompt(self, title: str, body: str) -> str:
        """プロンプトテンプレートに基づいて入力を構築する"""
        templates = {
            0: f"タイトル: {title}\n本文: {body}\nラベル: ",
            1: f"タイトル: {title}\n本文: {body}",
            2: f"{title}\n{body}",
        }
        return templates.get(self.args.template_type, f"{title}\n{body}")

    def _create_optimizer(self, steps_per_epoch: int):
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
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

    def _prepare_batch_for_model(self, batch: dict) -> dict:
        """モデルに渡す前に不要なキーを削除する"""
        # token_type_ids は多くのCausal LMで不要
        if "token_type_ids" in batch:
            batch.pop("token_type_ids")
        return batch

    def run_training(self):
        """訓練と評価のメインループを実行する"""
        best_val_f1, best_epoch = 0.0, -1
        best_state_dict = None

        # --- 訓練開始前の初期評価 ---
        initial_metrics = self._evaluate_step("valid", self.val_dataloader)
        self._log_metrics({"epoch": -1, **initial_metrics})

        # --- 訓練ループ ---
        for epoch in trange(self.args.epochs, desc="Epochs", dynamic_ncols=True):
            epoch_start_time = time.perf_counter()

            # 訓練ステップ
            train_metrics = self._train_epoch()
            train_metrics["train.elapsed_time"] = time.perf_counter() - epoch_start_time

            # 評価ステップ
            val_metrics = self._evaluate_step("valid", self.val_dataloader)

            # メトリクスの集約とロギング
            metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            self._log_metrics(metrics)

            # ベストモデルの保存
            if metrics["valid.f1"] > best_val_f1:
                best_val_f1 = metrics["valid.f1"]
                best_epoch = epoch
                # unwrap_model で Accelerator のラッパーを解除してから state_dict を取得
                best_state_dict = self.accelerator.get_state_dict(
                    self.accelerator.unwrap_model(self.model)
                )
                tqdm.write(
                    f"🎉 New best model found at epoch {epoch} "
                    f"with F1: {best_val_f1:.4f}"
                )

        # --- 訓練終了後の最終評価 ---
        tqdm.write(f"Best model from epoch {best_epoch} with F1: {best_val_f1:.4f}")
        if best_state_dict:
            # ベストモデルの重みをロード
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(best_state_dict)

            # 検証・テストデータで最終評価
            best_val_metrics = self._evaluate_step("best.valid", self.val_dataloader)
            best_test_metrics = self._evaluate_step("best.test", self.test_dataloader)

            final_metrics = {
                "best.epoch": best_epoch,
                **best_val_metrics,
                **best_test_metrics,
            }
            self._log_metrics(final_metrics)
            utils.save_json(final_metrics, self.args.output_dir / "final-metrics.json")

            # モデルの保存
            self.accelerator.save_state(output_dir=self.args.output_dir / "best_model")

    def _train_epoch(self) -> dict:
        """1エポック分の訓練を実行する"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(
            self.train_dataloader, desc="Training", dynamic_ncols=True, leave=False
        )
        for batch in pbar:
            with self.accelerator.accumulate(self.model):
                _batch = self._prepare_batch_for_model(batch)
                outputs: SequenceClassifierOutput = self.model(**_batch)
                loss = outputs.loss

                # 損失を収集して表示
                total_loss += loss.detach().float()
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

        # エポック全体の平均損失を計算
        avg_epoch_loss = self.accelerator.gather(total_loss).mean().item() / len(
            self.train_dataloader
        )
        return {"train.loss": avg_epoch_loss}

    @torch.inference_mode()
    def _evaluate_step(self, prefix: str, dataloader: DataLoader) -> dict:
        """指定されたデータローダーで評価を実行し、メトリクスを返す"""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0

        for batch in tqdm(
            dataloader, desc=f"Evaluating {prefix}", dynamic_ncols=True, leave=False
        ):
            _batch = self._prepare_batch_for_model(batch)
            outputs: SequenceClassifierOutput = self.model(**_batch)

            total_loss += outputs.loss.detach().float()

            preds = outputs.logits.argmax(dim=-1)
            labels = _batch["labels"]

            # Accelerator でプロセス間のテンソルを収集
            all_preds.append(self.accelerator.gather(preds))
            all_labels.append(self.accelerator.gather(labels))

        # 収集したテンソルをCPUに移動し、リストに変換
        all_preds = torch.cat(all_preds).cpu().tolist()
        all_labels = torch.cat(all_labels).cpu().tolist()

        # データセット全体で切り捨てられていないか確認
        if len(all_preds) > len(dataloader.dataset):
            all_preds = all_preds[: len(dataloader.dataset)]
            all_labels = all_labels[: len(dataloader.dataset)]

        # メトリクス計算
        avg_loss = total_loss.mean().item() / len(dataloader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)

        return {
            f"{prefix}.loss": avg_loss,
            f"{prefix}.accuracy": accuracy,
            f"{prefix}.precision": precision,
            f"{prefix}.recall": recall,
            f"{prefix}.f1": f1,
        }

    def _log_metrics(self, metrics: dict):
        """コンソールと MLflow にメトリクスを記録する"""
        if not self.accelerator.is_main_process:
            return

        epoch = metrics.get("epoch", -1)
        # MLflow にロギング
        mlflow.log_metrics(metrics, step=epoch if epoch != -1 else None)

        # CSVファイルにロギング
        # epoch をキーに持つログのみを記録
        if "epoch" in metrics:
            utils.log(metrics, self.args.output_dir / "log.csv")

        # コンソールに表示
        log_str = f"Epoch: {metrics.get('epoch', 'N/A'):>2} |"
        if "train.loss" in metrics:
            log_str += f" Train Loss: {metrics['train.loss']:.4f} |"
        if "valid.f1" in metrics:
            log_str += f" Valid F1: {metrics['valid.f1']:.4f} |"
        if "valid.loss" in metrics:
            log_str += f" Valid Loss: {metrics['valid.loss']:.4f}"

        tqdm.write(log_str)


def main(args: Args):
    # MLflow のセットアップ
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("train")
    mlflow.system_metrics.enable_system_metrics_logging()
    mlflow.start_run()
    mlflow.log_params(args.as_dict())

    # 乱数シードの初期化
    utils.init(seed=args.seed)

    # トレーナーの初期化と実行
    trainer = Trainer(args)
    trainer.run_training()

    # configの保存
    utils.save_config(args, args.output_dir / "config.json")
    mlflow.end_run()


if __name__ == "__main__":
    cli_args = Args().parse_args()
    main(cli_args)
