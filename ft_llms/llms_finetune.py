import argparse

from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig, IA3Config
from transformers.utils import logging
logging.set_verbosity_info()
import sys

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from huggingface_hub import login
# login(token="")

# DEBUG: Print current file path to verify where we are
print(f"[DEBUG] This file is: {__file__}")


# Dynamically inject the real project root: up 2 levels
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"[DEBUG] sys.path injected project root: {PROJECT_ROOT}")

from data.prepare import dataset_prepare
from attack.utils import create_folder
from transformers import LlamaTokenizer, get_scheduler
from utils import get_logger, constantlengthdatasetiter, print_trainable_parameters
# trl.trainer.ConstantLengthDataset.__dict__["__iter__"] = constantlengthdatasetiter
# setattr(trl.trainer.ConstantLengthDataset, "__iter__", constantlengthdatasetiter)

logger = get_logger("finetune", "info")

# CLI argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="D:/documents/python codes/Thesis/github-code/ANeurIPS2024_SPV-MIA-main/ANeurIPS2024_SPV-MIA-main/deepseek-ai/deepseek-llm-7b-base")

    parser.add_argument("-d", "--dataset_name", type=str, default="D:/documents/python codes/Thesis/github-code/ANeurIPS2024_SPV-MIA-main/ANeurIPS2024_SPV-MIA-main/cache")

    parser.add_argument("-dc", "--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--cache_path", type=str, default="./cache")
    parser.add_argument("--use_dataset_cache", action="store_true", default=False)
    parser.add_argument("--refer", action="store_true", default=False)
    parser.add_argument("--refer_data_source", type=str, default=None)
    parser.add_argument("--packing", action="store_true", default=False)
    parser.add_argument("-t", "--token", type=str, default=None)
    parser.add_argument("--split_model", action="store_true", default=False)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--peft", type=str, default="lora")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--p_tokens", type=int, help="The number of virtual tokens for prefix-tuning or p-tuning", default=20)
    parser.add_argument("--p_hidden", type=int, help="The hidden size of the prompt encoder", default=128)

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--output_dir", type=str, default="/ft_llms/checkpoints")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)

    parser.add_argument("--train_sta_idx", type=int, default=0)
    parser.add_argument("--train_end_idx", type=int, default=6000)
    parser.add_argument("--eval_sta_idx", type=int, default=0)
    parser.add_argument("--eval_end_idx", type=int, default=600)

    parser.add_argument("-s", "--save_limit", type=int, default=None)

    parser.add_argument("--use_int4", action="store_true", default=False)
    parser.add_argument("--use_int8", action="store_true", default=False)
    parser.add_argument("--disable_peft", action="store_true", default=False)

    parser.add_argument("--pad_token_id", default=None, type=int, help="The end of sequence token.")
    parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token to tokenizer", default=False)
    parser.add_argument("--add_bos_token", action="store_true", help="Add BOS token to tokenizer", default=False)
    parser.add_argument("--validation_split_percentage", default=0.1, help="The percentage of the train set used as validation set in case there's no validation split")
    args = parser.parse_args()

    # Initialize accelerator for multi-GPU
    accelerator = Accelerator()

    if args.token is None:
        access_token = os.getenv("HF_TOKEN", "")
    else:
        access_token = args.token

    # Load model config
    config = AutoConfig.from_pretrained(args.model_name,cache_dir=args.cache_path,trust_remote_code=True)

    config.use_cache = False
    config_dict = config.to_dict()
    model_type = config_dict.get("model_type", "")



    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {"device_map": None}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True )



    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    # good one for LLama is 18610
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    block_size = args.block_size
    logger.info("Using a block size of %d", block_size)

    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        optimizer = "adamw_bnb_8bit"
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"

    # PEFT configuration
    if args.peft == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    elif args.peft == "prefix-tuing":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=args.p_tokens,
            encoder_hidden_size=args.p_hidden)
    elif args.peft == "p-tuing":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.p_tokens,
            encoder_hidden_size=args.p_hidden)
    elif args.peft == "ia3":
        peft_config = IA3Config(
            peft_type="IA3",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(config, 'quantization_config'):
        del config.quantization_config

    model = AutoModelForCausalLM.from_pretrained(
         args.model_name,
         trust_remote_code=True,
         cache_dir=args.cache_path,
         torch_dtype=torch_dtype,
         config=config,
         quantization_config=bnb_config,  # Must be **after** trust_remote_code
         ignore_mismatched_sizes=True,
         **kwargs
    )



    if not args.disable_peft:
        logger.info("Using PEFT...")
        if args.use_int4 or args.use_int8:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(model)
        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)
    else:
        logger.info("Using Full Finetuning")

    print_trainable_parameters(model)

    # Load datasets
    if args.refer_data_source is not None:
        args.model_name = args.refer_data_source
    if args.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
        args.model_name = "decapoda-research/llama-7b-hf"
    with accelerator.main_process_first():
        train_dataset, valid_dataset = dataset_prepare(args, tokenizer=tokenizer)
        if args.refer:
            train_dataset = None
            refer_data_path = f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/refer@{args.model_name}/"
            train_dataset = load_from_disk(refer_data_path)
        train_dataset = Dataset.from_dict(train_dataset[args.train_sta_idx:args.train_end_idx])
        valid_dataset = Dataset.from_dict(valid_dataset[args.eval_sta_idx:args.eval_end_idx])
        # train_dataset = load_from_disk("/mnt/data0/fuwenjie/MIA-LLMs/cache/ag_news/None/refer@gpt2")

    logger.info(f"Training with {Accelerator().num_processes} GPUs")

    # Set up training arguments
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_epochs,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        optim=optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=args.weight_decay,
        adam_epsilon=1e-6,
        report_to="wandb",
        load_best_model_at_end=False,
        save_total_limit=args.save_limit,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
    )

    # get trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )

    # train
    trainer.train()