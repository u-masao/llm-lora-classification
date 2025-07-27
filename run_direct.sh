#!/bin/bash

set -x

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 512
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 4096
exit
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name google/gemma-3-1b-pt
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-1.8b
exit

# CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name microsoft/Phi-4-mini-instruct
# CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name openai-community/gpt2
# CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-modernbert-base

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-0.6B
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-1.7B
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-440m
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-980m
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3.1-1.8b

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name cyberagent/open-calm-1b
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name cyberagent/open-calm-3b
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name cyberagent/open-calm-7b

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b-instruction-sft
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b-instruction-sft-v2
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b-instruction-ppo

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 0
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 1
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 2
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 3
