# Fix: https://github.com/unslothai/unsloth/issues/2298
from unsloth import FastLanguageModel
import logging
import os
import sys
import torch
import subprocess
from trl import GRPOConfig, GRPOTrainer, TrlParser
from utils.dataset import getCodingDataset
from utils.GRPOrewards import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)
import datasets
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
import transformers
max_seq_length = 2042 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# Initialize environment variables
from dotenv import load_dotenv
import os
load_dotenv()


logger = logging.getLogger(__name__)


def check_nvidia_smi():
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

import os


# def init_wandb_training(training_args):
#     """
#     Helper function for setting up Weights & Biases logging tools.
#     """
#     if training_args.wandb_entity is not None:
#         os.environ["WANDB_ENTITY"] = training_args.wandb_entity
#     if training_args.wandb_project is not None:
#         os.environ["WANDB_PROJECT"] = training_args.wandb_project
#     if training_args.wandb_run_group is not None:
#         os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group



def init_wandb_training():
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    os.environ["WANDB_ENTITY"] = "alejandro-paredeslatorre-duke-university"
    os.environ["WANDB_PROJECT"] = "qwen-cot-training-qwen2.5-0.5B-v2"
    os.environ["WANDB_RUN_GROUP"] = "qwen1.experiment_2"

#def main(script_args, training_args, model_args):
def main():
    check_nvidia_smi()
    # Set seed for reproducibility
    # set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # if script_args and training_args and model_args:
    #     # Log on each process a small summary
    #     logger.warning(
    #         f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #         + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    #     )
    #     logger.info(f"Model parameters {model_args}")
    #     logger.info(f"Script parameters {script_args}")
    #     logger.info(f"Training parameters {training_args}")

    # # Check for last checkpoint
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # if "wandb" in training_args.report_to:
    #     init_wandb_training(training_args)
    
    # from transformers import AutoTokenizer
    # tokenizer_fixed = AutoTokenizer.from_pretrained("meta-llama/meta-Llama-3.1-8B-Instruct", token=os.getenv("HF_TOKEN"))
    # tokenizer_fixed.pad_token = tokenizer_fixed.eos_token
    # tokenizer_fixed.padding_side = "left"
    # tokenizer_fixed.truncation_side = "left"
    # tokenizer_fixed.truncation = True
    # tokenizer_fixed.model_max_length = max_seq_length
    init_wandb_training()
    #os.environ["HF_HOME"] = "/home/users/ap794/.cache/huggingface"
    model, tokenizer = FastLanguageModel.from_pretrained(
        #model_name = '/home/users/ap794/final_project_distillLLM/minillm/results/qwen2.5/train/sft/qwen2.5-0.5B-Instruct/e10-bs1-lr1e-05-G2-N2-NN1/8000',
        #model_name = '/home/users/ap794/final_project_distillLLM/minillm/results/qwen2.5/train/sft/qwen2.5-1.5B-Instruct/e10-bs1-lr1e-05-G2-N4-NN1/8000',
        model_name = "/home/users/ap794/final_project_distillLLM/minillm/results/qwen2.5/train/kd/Qwen2.5-0.5B-to-Qwen2.5-1.5B-sft/e10-bs8-lr1e-05-G1-N2-NN1-kd0.5/4000",
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, # Reduce if out of memory
        token=os.getenv("HF_TOKEN"),
    )
    check_nvidia_smi()
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )
    max_prompt_length = 512
    check_nvidia_smi()
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 6, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 5000,
        save_steps = 500,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = "outputs",
    )
    # Preprocess your datasets
    # train_dataset, test_dataset = getCodingDataset("train", tokenizer)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset=getCodingDataset("train")
        # train_dataset=train_dataset,
        # eval_dataset=test_dataset,
    )
    check_nvidia_smi()

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    # train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.train()
    # metrics = train_result.metrics
    #metrics["train_samples"] = len(getCodingDataset("train")[script_args.dataset_train_split])
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    trainer.save_state()
    check_nvidia_smi()
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    check_nvidia_smi()
    
    # # Save everything else on main process
    # kwargs = {
    #     "dataset_name": script_args.dataset_name,
    #     "tags": ["open-r1"],
    # }
    # if trainer.accelerator.is_main_process:
    #     trainer.create_model_card(**kwargs)
    #     # Restore k,v cache for fast inference
    #     trainer.model.config.use_cache = True
    #     trainer.model.config.save_pretrained(training_args.output_dir)

    # ##########
    # # Evaluate
    # ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # #############
    # # push to hub
    # #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    print("Running GRPO script")
    # parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    # script_args, training_args, model_args = parser.parse_args_and_config()
    main()