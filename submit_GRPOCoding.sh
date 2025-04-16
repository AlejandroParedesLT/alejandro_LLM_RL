#!/bin/bash
#SBATCH -t 5:00:00  # time requested in hour:minute:second
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000 #24 a6000, v100, a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out

# Load environment variables from .env
set -a
source $HOME/final_project_distillLLM/open-r1/.env
set +a

# Login to Hugging Face CLI without affecting Git
echo "$HF_TOKEN" | huggingface-cli login --token --no-git
if [ $? -eq 0 ]; then
  echo "[✅] Successfully logged into Hugging Face"
else
  echo "[❌] Failed to log into Hugging Face"
fi

# Login to wandb
wandb login --relogin "$WANDB_API_KEY"
if [ $? -eq 0 ]; then
  echo "[✅] Successfully logged into Weights & Biases"
else
  echo "[❌] Failed to log into Weights & Biases"
fi

export HF_HOME=/dev/shm/hf-home
export TRANSFORMERS_CACHE=/dev/shm/hf-cache
export HF_DATASETS_CACHE=/dev/shm/hf-datasets
export TORCH_HOME=/dev/shm/torch-home
export XDG_CACHE_HOME=/dev/shm/.cache
export WANDB_CACHE_DIR=/dev/shm/wandb-cache

export VENV_DIR=$HOME/final_project_distillLLM/aleGRPO/grpo_venv

# srun bash -c "source $VENV_DIR/bin/activate &&  CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
#     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml"


srun bash -c "source $VENV_DIR/bin/activate && python src/main.py"