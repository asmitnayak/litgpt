#!/bin/bash

export WANDB_PROJECT_NAME="litgpt-dp-finetuning"

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --config config_hub/finetune/dp-finetuning/smollm2-1.7B/lora.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --config config_hub/finetune/dp-finetuning/smollm2-1.7B/lora-reordered.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora HuggingFaceTB/SmolLM2-135M-Instruct \
    --config config_hub/finetune/dp-finetuning/smollm2-135M/lora.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora HuggingFaceTB/SmolLM2-135M-Instruct \
    --config config_hub/finetune/dp-finetuning/smollm2-135M/lora-reordered.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora HuggingFaceTB/SmolLM2-360M-Instruct \
    --config config_hub/finetune/dp-finetuning/smollm2-360M/lora.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora HuggingFaceTB/SmolLM2-360M-Instruct \
    --config config_hub/finetune/dp-finetuning/smollm2-360M/lora-reordered.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora meta-llama/Llama-3.2-1B-Instruct \
    --config config_hub/finetune/dp-finetuning/llama3.2-1B/lora.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora meta-llama/Llama-3.2-1B-Instruct \
    --config config_hub/finetune/dp-finetuning/llama3.2-1B/lora-reordered.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora meta-llama/Llama-3.2-3B-Instruct \
    --config config_hub/finetune/dp-finetuning/llama3.2-3B/lora.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora meta-llama/Llama-3.2-3B-Instruct \
    --config config_hub/finetune/dp-finetuning/llama3.2-3B/lora-reordered.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora google/gemma-2-2b-it \
    --config config_hub/finetune/dp-finetuning/gemma2-2B/lora.yaml \
    --devices 6

# Delay for 30 seconds
sleep 30

NCCL_P2P_DISABLE=1 HF_TOKEN="$1" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
litgpt finetune_lora google/gemma-2-2b-it \
    --config config_hub/finetune/dp-finetuning/gemma2-2B/lora-reordered.yaml \
    --devices 6