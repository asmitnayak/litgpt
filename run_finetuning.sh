#!/bin/bash

echo -e "Running finetuning script...\n"
echo -e "Using HF_TOKEN: $1\n"
export WANDB_PROJECT_NAME="litgpt-dp-finetuning"
echo -e "WANDB_PROJECT_NAME: $WANDB_PROJECT_NAME\n"

# Ask user for confirmation
read -p "Do you want to continue running the script? (Y/n): " continue_script

# Check if user wants to continue (case-insensitive 'y' or 'Y')
if [[ "$continue_script" =~ ^[Yy]$ ]]; then
  echo -e "\nContinuing script...\n"

  # Common environment variables (set individually)
  export NCCL_P2P_DISABLE=1
  echo -e "setting NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
  export HF_TOKEN="$1"
  echo -e "setting HF_TOKEN=$1"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  echo -e "setting CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
  echo -e "setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES\n"
  DEVICES="--devices 6"
  LITGPT_CMD="litgpt finetune_lora"

  # Array of model and config pairs
  declare -a models_configs=(
    #"HuggingFaceTB/SmolLM2-1.7B-Instruct config_hub/finetune/dp-finetuning/smollm2-1.7B/lora.yaml"
    #"HuggingFaceTB/SmolLM2-1.7B-Instruct config_hub/finetune/dp-finetuning/smollm2-1.7B/lora-reordered.yaml"
    #"HuggingFaceTB/SmolLM2-135M-Instruct config_hub/finetune/dp-finetuning/smollm2-135M/lora.yaml"
    #"HuggingFaceTB/SmolLM2-135M-Instruct config_hub/finetune/dp-finetuning/smollm2-135M/lora-reordered.yaml"
    #"HuggingFaceTB/SmolLM2-360M-Instruct config_hub/finetune/dp-finetuning/smollm2-360M/lora.yaml"
    #"HuggingFaceTB/SmolLM2-360M-Instruct config_hub/finetune/dp-finetuning/smollm2-360M/lora-reordered.yaml"
    #"meta-llama/Llama-3.2-1B-Instruct config_hub/finetune/dp-finetuning/llama3.2-1B/lora.yaml"
    #"meta-llama/Llama-3.2-1B-Instruct config_hub/finetune/dp-finetuning/llama3.2-1B/lora-reordered.yaml"
    #"meta-llama/Llama-3.2-3B-Instruct config_hub/finetune/dp-finetuning/llama3.2-3B/lora.yaml"
    #"meta-llama/Llama-3.2-3B-Instruct config_hub/finetune/dp-finetuning/llama3.2-3B/lora-reordered.yaml"
    #"google/gemma-2-2b-it config_hub/finetune/dp-finetuning/gemma2-2B/lora.yaml"
    #"google/gemma-2-2b-it config_hub/finetune/dp-finetuning/gemma2-2B/lora-reordered.yaml"
    #"Qwen/Qwen2.5-0.5B-Instruct config_hub/finetune/dp-finetuning/qwen2.5-0.5B/lora.yaml"
    #"Qwen/Qwen2.5-0.5B-Instruct config_hub/finetune/dp-finetuning/qwen2.5-0.5B/lora-reordered.yaml"
    #"Qwen/Qwen2.5-1.5B-Instruct config_hub/finetune/dp-finetuning/qwen2.5-1.5B/lora.yaml"
    #"Qwen/Qwen2.5-1.5B-Instruct config_hub/finetune/dp-finetuning/qwen2.5-1.5B/lora-reordered.yaml"
    "Qwen/Qwen2.5-3B-Instruct config_hub/finetune/dp-finetuning/qwen2.5-3B/lora.yaml"
    #"Qwen/Qwen2.5-3B-Instruct config_hub/finetune/dp-finetuning/qwen2.5-3B/lora-reordered.yaml"
  )

  # Statement to confirm the number of models and configs
  echo -e "Running finetuning for ${#models_configs[@]} models and configs...\n"

  # Loop through models and configs
  for model_config in "${models_configs[@]}"; do
    IFS=' ' read -r model config <<< "$model_config" # Split model_config string into model and config variables

    # Execute the command directly - NO EVAL
    echo -e "Executing: $LITGPT_CMD $model --config $config $DEVICES\n"
    $LITGPT_CMD "$model" --config "$config" $DEVICES

    echo "$config" >> ./.done_configs
    # Delay for 10 seconds
    sleep 10

    # Timed input prompt
    read -t 15 -p "Continue to the next command? (Y/n, default is Yes): " continue_loop
    if [[ ! -z "$continue_loop" ]]; then # Check if input is NOT empty
      if [[ ! "$continue_loop" =~ ^[Yy]$ ]]; then # If input is NOT 'Y' or 'y' (case-insensitive)
        echo "User chose to stop or entered invalid input. Exiting loop."
        break # Exit the loop
      fi
    fi
    echo -e "\nContinuing to the next command...\n" # If no input or 'Y'/'y', continue
  done

else
  echo "Script execution cancelled by user."
  exit 0 # Exit script with success code (0)
fi

echo "Finetuning script completed."