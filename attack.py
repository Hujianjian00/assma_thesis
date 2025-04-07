import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import random
from pathlib import Path
from attack.attack_model import AttackModel
from data.prepare import dataset_prepare
from attack.utils import Dict

import yaml
import datasets
from datasets import Image, Dataset
from accelerate import Accelerator
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()


from accelerate.logging import get_logger
PATH = os.path.dirname(os.path.abspath(__file__))

# Set up logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("attack.log", mode="w")
    ]
)

from accelerate import Accelerator
from accelerate.logging import get_logger

# Initialize accelerator and logger in the global scope
accelerator = Accelerator()
logger = get_logger(__name__, "INFO")



# Load config file
with open("D:/documents/python codes/Thesis/github-code/ANeurIPS2024_SPV-MIA-main/ANeurIPS2024_SPV-MIA-main/configs/config.yaml", 'r') as f:

    cfg = yaml.safe_load(f)
    cfg = Dict(cfg)
    # Ensure cache and attack directories exist
    os.makedirs(cfg["cache_path"], exist_ok=True)
    os.makedirs(os.path.join(PATH, cfg["attack_data_path"]), exist_ok=True)

    cfg["safe_model_name"] = Path(cfg["model_name"]).stem
    cfg["safe_dataset_name"] = Path(cfg["dataset_name"]).stem
args.__dict__.update(cfg)
# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))

# Fix the random seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

## Load generation models.
if not cfg["load_attack_data"]:
    #config = AutoConfig.from_pretrained(cfg["model_name"])
    # config.use_cache = False
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Modify this section where the model is loaded
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        local_files_only=True,
        config=AutoConfig.from_pretrained(cfg["target_model"]),  # Use target_model path here
        cache_dir=cfg["cache_path"]
    )

    reference_model = AutoModelForCausalLM.from_pretrained(
        cfg["reference_model"],  # Same for reference model path
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,  # Same here, change if needed
        local_files_only=True,
        config=AutoConfig.from_pretrained(cfg["target_model"]),
        cache_dir=cfg["cache_path"]
    )

    logger.info("Successfully load models")
    config = AutoConfig.from_pretrained(cfg["target_model"])

    # Load tokenizer.
    model_type = config.to_dict()["model_type"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"],
                                            add_eos_token=cfg["add_eos_token"],
                                            add_bos_token=cfg["add_bos_token"],
                                            use_fast=True)


    if cfg["model_name"] == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
        cfg["model_name"] = "decapoda-research/llama-7b-hf"

    if cfg["pad_token_id"] is not None:
        logger.info("Using pad token id %d", cfg["pad_token_id"])
        tokenizer.pad_token_id = cfg["pad_token_id"]

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
    print(f"Train set size (member): {len(train_dataset)}")
    print(f"Validation set size (non-member): {len(valid_dataset)}")

    train_dataset = Dataset.from_dict(train_dataset[cfg.train_sta_idx:cfg.train_end_idx])
    valid_dataset = Dataset.from_dict(valid_dataset[cfg.eval_sta_idx:cfg.eval_end_idx])
    train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), cfg.maximum_samples))
    actual_len = len(valid_dataset)
    print(f"🟡 Non-member (valid) dataset actual size: {actual_len}")
    num_samples = min(cfg["maximum_samples"], actual_len)
    valid_dataset = valid_dataset.select(random.sample(range(len(valid_dataset)), cfg.maximum_samples))


    print(f"Valid dataset size before sampling: {len(valid_dataset)}")
    print(f"Requested maximum_samples: {cfg['maximum_samples']}")

    logger.info("Successfully load datasets!")

    # Prepare dataloade
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["eval_batch_size"])
    eval_dataloader = DataLoader(valid_dataset, batch_size=cfg["eval_batch_size"])

    # Load Mask-f
    shadow_model = None
    int8_kwargs = {}
    half_kwargs = {}
    if cfg["int8"]:
        int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif cfg["half"]:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["mask_filling_model_name"], **int8_kwargs, **half_kwargs).to(accelerator.device)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = AutoTokenizer.from_pretrained(cfg["mask_filling_model_name"], model_max_length=n_positions)

    # Prepare everything with accelerator
    train_dataloader, eval_dataloader = (
        accelerator.prepare(
            train_dataloader,
            eval_dataloader,
    ))
else:
    target_model = None
    reference_model = None
    shadow_model = None
    mask_model = None
    train_dataloader = None
    eval_dataloader = None
    tokenizer = None
    mask_tokenizer = None


datasets = {
    "target": {
        "train": train_dataloader,
        "valid": eval_dataloader
    }
}


attack_model = AttackModel(target_model, tokenizer, datasets, reference_model, shadow_model, cfg, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
attack_model.conduct_attack(cfg=cfg)


