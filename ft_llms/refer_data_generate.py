import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data.prepare import dataset_prepare
from attack.utils import Dict
import argparse
import yaml
import datasets
from datasets import Image, Dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig
from pathlib import Path
import os
os.environ['HTTP_PROXY'] = 'http://fuwenjie:19990621f@localhost:7890'
os.environ['HTTPS_PROXY'] = 'http://fuwenjie:19990621f@localhost:7890'

# Load config file
accelerator = Accelerator()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-tm", "--target_model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-d", "--dataset_name", type=str, default="wikitext-2-raw-v1")
parser.add_argument("-dc", "--dataset_config_name", type=str, default=None,
                    help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--cache_path", type=str, default="./cache")
parser.add_argument("--use_dataset_cache", action="store_true", default=True)
parser.add_argument("--packing", action="store_true", default=True)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--preprocessing_num_workers", type=int, default=1)
parser.add_argument("--validation_split_percentage", default=0.1,
                    help="The percentage of the train set used as validation set in case there's no validation split")
cfg = parser.parse_args()

print(accelerator.device)

config = AutoConfig.from_pretrained(cfg.model_name)
config.use_cache = False
bnb_config = None
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(cfg.target_model, quantization_config=bnb_config,
                                                    torch_dtype=torch_dtype,
                                                    local_files_only=True,
                                                    config=config,
                                                    cache_dir=cfg.cache_path)
model_type = config.to_dict()["model_type"]
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)


if tokenizer.pad_token_id is None:
    print("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
# Load datasets
if not hasattr(cfg, "dataset_path"):
    cfg.dataset_path = cfg.dataset_name  # fallback if dataset_path not defined
train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)

prompt_dataset = train_dataset.filter(lambda example: len(example["text"]) < 3000)


prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)


model, prompt_dataloader = accelerator.prepare(model, prompt_dataloader)

generated_dataset = {"text": []}

for text in tqdm(prompt_dataloader):
    prompt = (text["text"])
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(accelerator.device)
    attention_mask = inputs.attention_mask.to(accelerator.device)

    clipped_ids = input_ids[:, :16]
    if hasattr(model, "module"):
        gen_tokens = model.module.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    else:
        gen_tokens = model.generate(
            clipped_ids,
            attention_mask=attention_mask[:, :16],  # clip the same way as input_ids
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )

    if model_type == "llama":
        gen_tokens = gen_tokens[:, 1:]
    print(model(gen_tokens, labels=gen_tokens).loss)
    gen_text = tokenizer.batch_decode(gen_tokens)
    generated_dataset["text"].extend(gen_text)

generated_dataset = Dataset.from_dict(generated_dataset)
if cfg.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
    cfg.model_name = "decapoda-research/llama-7b-hf"
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data.prepare import dataset_prepare
from attack.utils import Dict
import argparse
import yaml
import datasets
from datasets import Image, Dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig
from pathlib import Path
import os
os.environ['HTTP_PROXY'] = 'http://fuwenjie:19990621f@localhost:7890'
os.environ['HTTPS_PROXY'] = 'http://fuwenjie:19990621f@localhost:7890'

# Load config file
accelerator = Accelerator()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-tm", "--target_model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-d", "--dataset_name", type=str, default="wikitext-2-raw-v1")
parser.add_argument("-dc", "--dataset_config_name", type=str, default=None,
                    help="The configuration name of the dataset to use (via the datasets library).")
parser.add_argument("--cache_path", type=str, default="./cache")
parser.add_argument("--use_dataset_cache", action="store_true", default=True)
parser.add_argument("--packing", action="store_true", default=True)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--preprocessing_num_workers", type=int, default=1)
parser.add_argument("--validation_split_percentage", default=0.1,
                    help="The percentage of the train set used as validation set in case there's no validation split")
cfg = parser.parse_args()

print(accelerator.device)

config = AutoConfig.from_pretrained(cfg.model_name)
config.use_cache = False
bnb_config = None
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(cfg.target_model, quantization_config=bnb_config,
                                                    torch_dtype=torch_dtype,
                                                    local_files_only=True,
                                                    config=config,
                                                    cache_dir=cfg.cache_path)
model_type = config.to_dict()["model_type"]
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)


if tokenizer.pad_token_id is None:
    print("Pad token id is None, setting to eos token id...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
# Load datasets
train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
prompt_dataset = train_dataset.select(range(min(len(train_dataset), 500)))

prompt_dataloader = DataLoader(prompt_dataset, batch_size=1)


model, prompt_dataloader = accelerator.prepare(model, prompt_dataloader)

generated_dataset = {"text": []}

for text in tqdm(prompt_dataloader):
    prompt = (text["text"])
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(accelerator.device)
    attention_mask = inputs.attention_mask.to(accelerator.device)

    clipped_ids = input_ids[:, :16]
    if hasattr(model, "module"):
        gen_tokens = model.module.generate(
            clipped_ids,
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )
    else:
        gen_tokens = model.generate(
            clipped_ids,
            attention_mask=attention_mask[:, :16],  # clip the same way as input_ids
            num_beams=1,
            do_sample=True,
            max_length=input_ids.size(-1),
        )

    if model_type == "llama":
        gen_tokens = gen_tokens[:, 1:]
    print(model(gen_tokens, labels=gen_tokens).loss)
    gen_text = tokenizer.batch_decode(gen_tokens)
    generated_dataset["text"].extend(gen_text)

generated_dataset = Dataset.from_dict(generated_dataset)
if cfg.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
    cfg.model_name = "decapoda-research/llama-7b-hf"
safe_dataset_name = Path(cfg.dataset_name).stem
safe_model_name = Path(cfg.model_name).stem

save_dir = f"{cfg.cache_path}/{safe_dataset_name}/{cfg.dataset_config_name or 'default'}/refer@{safe_model_name}/"
os.makedirs(save_dir, exist_ok=True)

generated_dataset.save_to_disk(save_dir + f"{accelerator.device}")

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    concatenated_dataset = None
    for sub_dir in os.listdir(save_dir):
        data_path = os.path.join(save_dir, sub_dir)
        if os.path.isdir(data_path):
            if concatenated_dataset is None:
                concatenated_dataset = load_from_disk(data_path)
            else:
                dataset = load_from_disk(data_path)
                concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])
    concatenated_dataset.save_to_disk(save_dir)
save_dir = f"{cfg.cache_path}/{safe_dataset_name}/{cfg.dataset_config_name or 'default'}/refer@{safe_model_name}/"
os.makedirs(save_dir, exist_ok=True)
generated_dataset.save_to_disk(save_dir + f"{accelerator.device}")

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    concatenated_dataset = None
    for sub_dir in os.listdir(save_dir):
        data_path = os.path.join(save_dir, sub_dir)
        if os.path.isdir(data_path):
            if concatenated_dataset is None:
                concatenated_dataset = load_from_disk(data_path)
            else:
                dataset = load_from_disk(data_path)
                concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])
    concatenated_dataset.save_to_disk(save_dir)