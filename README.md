# alejandro_LLM_RL

# GRPO Fine-tuning for Llama 3.1 using Unsloth

This repository contains code for fine-tuning the Meta-Llama 3.1 8B Instruct model using Greedy Reward Policy Optimization (GRPO) with the Unsloth library for accelerated training.

## Overview

This project implements GRPO fine-tuning for improving code generation capabilities with multiple reward functions. It uses the Unsloth library to accelerate training by optimizing memory usage and enabling faster inference.

## Features

- Fine-tunes Meta-Llama 3.1 8B Instruct model
- Uses GRPO (Greedy Reward Policy Optimization) for training
- Implements multiple code-specific reward functions
- Accelerates training with Unsloth's optimizations
- Supports 4-bit quantization for efficient memory usage
- Integrates with Weights & Biases for experiment tracking

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU
- Hugging Face account with access to Meta-Llama 3.1 models

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/grpo-llama-unsloth.git
cd grpo-llama-unsloth

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file with your Hugging Face token:
```
HF_TOKEN=your_hugging_face_token_here
```

## Project Structure

- `main.py` - The main training script
- `utils/dataset.py` - Dataset preparation utilities
- `utils/GRPOrewards.py` - Custom reward functions for code generation

## Reward Functions

The model is trained with multiple reward functions specifically designed for code generation:

1. `xmlcount_reward_func` - Rewards proper XML tag usage
2. `soft_format_reward_func` - Evaluates formatting quality with flexible criteria
3. `strict_format_reward_func` - Enforces strict formatting requirements
4. `int_reward_func` - Rewards correct integer handling
5. `correctness_reward_func` - Evaluates overall code correctness

## Usage

```bash
python main.py
```

## Configuration

The script uses the following key configuration parameters:

- `max_seq_length`: 2042 tokens
- `lora_rank`: 32 (Higher values = more capacity but slower training)
- Quantization: 4-bit loading for memory efficiency
- Learning rate: 5e-6 with cosine scheduler
- Training steps: 250 with checkpoints saved every 250 steps

## Memory Optimization

The script includes several memory optimization techniques:

- 4-bit quantization
- Gradient checkpointing
- vLLM fast inference
- Configurable GPU memory utilization

## Weights & Biases Integration

Training progress can be monitored with Weights & Biases. Set your W&B configuration in the environment variables:

```
WANDB_ENTITY=your_entity
WANDB_PROJECT=your_project
WANDB_RUN_GROUP=your_run_group
```

## Issues

If you encounter any issues, please check if they're related to the fixed issue in:
https://github.com/unslothai/unsloth/issues/2298

## License

[Your chosen license]

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for the optimization library
- [TRL](https://github.com/huggingface/trl) for the GRPO implementation