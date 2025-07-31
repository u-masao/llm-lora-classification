#!/bin/bash

set -x

CUDA_VISIBLE_DEVICES=0

MODELS=(llm-jp/llm-jp-3-150m Qwen/Qwen3-0.6B google/gemma-3-1b-pt)
# MODELS=(llm-jp/llm-jp-3-150m)
SEQ_LENGTHS=(5500)
EPOCHS=10

for MODEL in "${MODELS[@]}"; do
    for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
        CMD="uv run python src/train.py --model_name $MODEL \
                --max_seq_len $SEQ_LENGTH --epochs $EPOCHS"
        $CMD
    done
done

exit

uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 40 --epochs 10 --use_unsloth --use_output_hidden_states
uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 40 --epochs 10 --use_unsloth
uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 40 --epochs 10 --use_output_hidden_states
uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 40 --epochs 10
uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 4000 --epochs 10 --use_unsloth --use_output_hidden_states
uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 4000 --epochs 10 --use_unsloth
uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 4000 --epochs 10 --use_output_hidden_states
uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 4000 --epochs 10
exit

SEQ_LENGTHS=(512 1024 2048 3072 4096 6144 8192)
MODEL=llm-jp/llm-jp-3-150m

for SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    CMD="uv run python src/train.py --model_name $MODEL --max_seq_len $SEQ_LENGTH"
    $CMD
    $CMD --use_unsloth
done

exit

CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 2048
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 4096
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 8192
exit

exit
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 512
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m --max_seq_len 4096
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-440m --max_seq_len 4096
exit
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-980m --max_seq_len 4096
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-0.6B --max_seq_len 4096
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-1.7B
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name google/gemma-3-1b-pt --max_seq_len 4096
exit

# trouble
# CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name microsoft/Phi-4-mini-instruct
# CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name openai-community/gpt2
# CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-modernbert-base

# Qwen
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-0.6B
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-1.7B
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name Qwen/Qwen3-4B

# llm-jp
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-150m
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-440m
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-980m
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --model_name llm-jp/llm-jp-3-1.8b
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
