# src/models.py

import peft
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutput,
)


class ClassificationModel(nn.Module):
    """
    PEFT (LoRA) を適用した事前学習済みモデルに分類ヘッドを追加したモデル。
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lora_r: int,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        # 1. バックボーンとなる事前学習済みモデルをロード
        backbone: PreTrainedModel = AutoModel.from_pretrained(
            model_name,
            # BF16が利用可能なら自動的に使用
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None,
        )

        # 2. LoRA (PEFT) の設定を適用
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            inference_mode=False,
            # モデルによっては target_modules の指定が必要な場合がある
            # target_modules=["query_key_value"],
        )
        self.backbone: PeftModel = peft.get_peft_model(backbone, peft_config)

        if gradient_checkpointing:
            self.backbone.enable_input_require_grads()
            self.backbone.gradient_checkpointing_enable()

        # 3. 分類ヘッドと損失関数を定義
        hidden_size: int = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor,
        labels: LongTensor = None,
    ) -> SequenceClassifierOutput:
        """
        フォワードパス。
        最終層の隠れ状態のうち、各系列の最後のトークンに対応するベクトルを用いて分類を行う。
        """
        # バックボーンモデルから隠れ状態を取得
        outputs: BaseModelOutputWithPast = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state

        # 各系列の最後のトークン（パディングを除く）の隠れ状態を抽出
        seq_lengths = attention_mask.sum(dim=1)
        eos_hidden_states = last_hidden_state[
            torch.arange(last_hidden_state.size(0), device=last_hidden_state.device),
            seq_lengths - 1,
        ]

        # 分類ヘッドを通してロジットを計算
        logits: FloatTensor = self.classifier(
            eos_hidden_states.to(self.classifier.weight.dtype)
        )

        # ラベルが与えられていれば損失を計算
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
