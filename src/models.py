import mlflow
import peft
import torch
import torch.nn as nn
import torchinfo
from peft import LoraConfig, PeftModel
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutput,
)


class Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lora_r: int,
        max_seq_len: int = 128,
        gradient_checkpointing: bool = True,
        model_print_depth: int = 4,
    ):
        super().__init__()

        self._init_model_by_transformers(
            model_name=model_name,
            lora_r=lora_r,
            max_seq_len=max_seq_len,
            gradient_checkpointing=gradient_checkpointing,
        )

        hidden_size: int = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        # print model summary
        batch_size = 1
        model_summary = torchinfo.summary(
            self.backbone,
            depth=model_print_depth,
            input_size=[
                (batch_size, max_seq_len),
                (batch_size, max_seq_len),
                (batch_size, max_seq_len),
            ],
            dtypes=[torch.long, torch.long, torch.long],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=15,
            verbose=0,
        )
        mlflow.log_text(str(model_summary), "model_summary.txt")
        mlflow.log_metrics(
            {
                "model.total_params": model_summary.total_params,
                "model.trainable_params": model_summary.trainable_params,
            }
        )

    def _init_model_by_transformers(
        self,
        model_name: str,
        lora_r: int,
        max_seq_len: int,
        gradient_checkpointing: bool = True,
    ):
        backbone: PreTrainedModel = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None,
        )

        self.backbone: PeftModel = peft.get_peft_model(
            backbone,
            LoraConfig(
                r=lora_r,
                lora_alpha=16,
                lora_dropout=0.1,
                inference_mode=False,
            ),
        )

        if gradient_checkpointing:
            self.backbone.enable_input_require_grads()
            self.backbone.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor = None,
        labels: LongTensor = None,
    ) -> SequenceClassifierOutput:
        # take model
        model = self.backbone.model

        # take peft backbone output
        outputs: BaseModelOutputWithPast = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # pickup last hidden layer
        last_hidden_state = outputs.last_hidden_state

        # pickup last hidden feature
        seq_length: LongTensor = attention_mask.sum(dim=1)
        eos_hidden_states: FloatTensor = last_hidden_state[
            torch.arange(
                seq_length.size(0),
                device=last_hidden_state.device,
            ),
            seq_length - 1,
        ]

        # make logits
        logits: FloatTensor = self.classifier(eos_hidden_states.to(torch.float32))

        # make loss
        loss = None
        if labels is not None:
            loss: FloatTensor = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def write_trainable_params(self) -> None:
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        percentage = 100 * trainable_params / all_param
        all_param /= 1000000000
        trainable_params /= 1_000_000

        print(
            f"trainable params: {trainable_params:.2f}M || "
            f"all params: {all_param:.2f}B || "
            f"trainable%: {percentage:.4f}"
        )

    def clone_state_dict(self) -> dict:
        return {
            "backbone": peft.get_peft_model_state_dict(self.backbone),
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        peft.set_peft_model_state_dict(self.backbone, state_dict["backbone"])
        self.classifier.load_state_dict(state_dict["classifier"])
