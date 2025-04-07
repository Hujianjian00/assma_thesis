import os
import random
import datasets
import trl
from attack.utils import create_folder

from pathlib import Path
import json

from sympy import false

block_size = None
tokenizer_ = None
max_buff_size = None
text_column = None

def packing_texts(examples):
    more_examples = True
    packed_texts = []
    packed_ids = []
    # for key in examples.keys():
    assert list(examples.keys()) == ["text"], f"Expected keys ['text'], got {list(examples.keys())}"
    iterator = iter(examples["text"])
    # for sentence in examples["text"]:
    total_num = 0
    drop_num = 0
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                new_text = next(iterator)
                if len(new_text) > 10000:  #  Step 2: skip absurdly long strings
                    continue
                buffer.append(new_text)
                buffer_len += len(new_text)
            except StopIteration:
                more_examples = False
                break

        tokenized = tokenizer_(buffer, truncation=False)
        if not tokenized["input_ids"]:
            return {"text": []}  # Skip this batch if nothing survived
        tokenized_inputs = tokenized["input_ids"]

        inputs = tokenizer_.batch_decode(tokenized_inputs)
        tokenized_inputs = tokenizer_(inputs, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input)
        for i in range(0, len(all_token_ids), block_size):
            input_ids = all_token_ids[i: i + block_size]
            if len(input_ids) == block_size:
                packed_ids.append(input_ids)
                input_text = tokenizer_.decode(input_ids)
                total_num += 1
                if len(tokenizer_.encode(input_text)) == block_size:
                    packed_texts.append(input_text)
                    drop_num += 1
    print(f"Total examples: {total_num}, dropped num: {drop_num}, dropped rate: {1 - drop_num/total_num}")
    print(f"Total examples: {total_num}, dropped num: {drop_num}, drop rate: {1 - drop_num / total_num:.2%}")

    return {
        "text": packed_texts
    }
def dataset_prepare(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    import json
    if hasattr(args, "dataset_path") and args.dataset_path and Path(args.dataset_path).suffix == ".parquet":

        if hasattr(args, "member_indices_path") and os.path.isfile(args.member_indices_path):
            with open(args.member_indices_path, "r") as f:
                member_records = json.load(f)

            # Step 1: Load full raw dataset
            raw_datasets = datasets.load_dataset("parquet", data_files={"train": args.dataset_path})["train"]
            # Dynamically rename the correct column to "text"
            if "message" in raw_datasets.column_names:
                raw_datasets = raw_datasets.rename_column("message", "text")
            elif "file" in raw_datasets.column_names:
                raw_datasets = raw_datasets.rename_column("file", "text")

            # Step 2: Extract indices by checking presence in raw dataset
            all_texts = list(raw_datasets["text"])

            member_indices = set(member_records)
            non_member_indices = list(set(range(len(raw_datasets))) - member_indices)

            # Limit non-member count to maximum_samples
            if hasattr(args, "maximum_samples"):
                non_member_indices = non_member_indices[:args.maximum_samples]

            print(f"[INFO] Matched {len(member_indices)} member samples.")
            print(f"[INFO] Using {len(non_member_indices)} non-member samples.")

            train_dataset = raw_datasets.select(sorted(list(member_indices)))
            valid_dataset = raw_datasets.select(sorted(non_member_indices))

        else:
            raise ValueError("Missing or invalid member_indices_path.")

    else:

        train_dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{int((1 - args.validation_split_percentage) * 100)}%]"
        )
        valid_dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{int((1 - args.validation_split_percentage) * 100)}%:]"
        )

    # train_idxs = set(random.sample(range(len(raw_datasets)), int(len(raw_datasets) * (1 - args.validation_split_percentage))))
    # valid_idxs = set(range(len(raw_datasets))) - train_idxs
    # train_dataset = datasets.Dataset.from_dict(raw_datasets[train_idxs])
    # valid_dataset = datasets.Dataset.from_dict(raw_datasets[valid_idxs])


    global text_column
    column = train_dataset.column_names
    if "text" in column:
        text_column = "text"
    elif "document" in column:
        text_column = "document"
    elif "content" in column:
        text_column = "content"
    elif "message" in column:  # ← Add this line
        text_column = "message"

    train_dataset = train_dataset.select_columns(text_column)
    valid_dataset = valid_dataset.select_columns(text_column)
    if text_column != "text":
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")

    if args.packing:
        global block_size, tokenizer_, max_buff_size
        block_size = args.block_size
        max_buff_size = block_size * chars_per_token * num_of_sequences
        tokenizer_ = tokenizer
        safe_dataset_name = Path(args.dataset_path).stem if hasattr(args, "dataset_path") and args.dataset_path else args.dataset_name

        create_folder(f"{args.cache_path}/{safe_dataset_name}/{args.dataset_config_name}")

        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{safe_dataset_name}/{args.dataset_config_name}/train_dataset",


            load_from_cache_file=False,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{safe_dataset_name}/{args.dataset_config_name}/valid_dataset",

            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        return train_dataset, valid_dataset
    else:
        # Ensure datasets are returned even if packing is disabled
        return train_dataset, valid_dataset