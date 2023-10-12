from datasets import load_dataset, DatasetDict


def get_dataset(tokenizer, context_length, seed=42, train_size=100):
    ds_train = load_dataset(
        "huggingface-course/codeparrot-ds-train",
        split="train",
        cache_dir="/nfs/scratch_2/christian_wallenwein/.cache/huggingface/datasets",
    )
    ds_valid = load_dataset(
        "huggingface-course/codeparrot-ds-valid",
        split="validation",
        cache_dir="/nfs/scratch_2/christian_wallenwein/.cache/huggingface/datasets",
    )

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle(seed=42).select(range(train_size)),
            "valid": ds_valid.select(range(256)),
        }
    )

    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )

        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_dataset = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    return tokenized_dataset
