import sympy, sys
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments,  DataCollatorForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import TensorDataset
from datasets import load_dataset
import pandas as pd
import re

raw_entities = ["EMAIL", "EMAIL+ORGANIZATION", "NAME", "ORGANIZATION"]
entity_types = ["O"] + [f"B-{e}" for e in raw_entities] + [f"I-{e}" for e in raw_entities]
num_labels = len(entity_types)

# 3) Load the flashcards
dataset = load_dataset("Yijia-Xiao/pii-medical_flashcards", split="train")

# 4) Load a Fast tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", use_fast=True)

# 5) One‐pass preprocess: extract spans, tokenize, align labels
splitter = re.compile(r'(\{\{.*?\}\})')

def preprocess_function(batch):
    # a) tokenize with offsets
    encodings = tokenizer(
        batch["output"],
        padding=False,
        truncation=True,
        max_length=64,
        return_offsets_mapping=True
    )

    all_labels = []
    for orig, masked, offsets in zip(batch["output"],
                                     batch["cleaned_output"],
                                     encodings["offset_mapping"]):
        # b) extract (start, end, ent) spans
        spans = []
        if isinstance(masked, str):
            segments = splitter.split(masked)
            idx = 0
            for i, seg in enumerate(segments):
                if seg.startswith("{{") and seg.endswith("}}"):
                    ent = seg[2:-2]
                    next_plain = segments[i+1] if i+1 < len(segments) else ""
                    if next_plain:
                        j = orig.find(next_plain, idx)
                        spans.append((idx, j, ent))
                        idx = j
                    else:
                        spans.append((idx, len(orig), ent))
                        idx = len(orig)
                else:
                    if seg:
                        j = orig.find(seg, idx)
                        idx = (j + len(seg)) if j != -1 else (idx + len(seg))

        # c) align tokens → labels
        labels = [entity_types.index("O")] * len(offsets)
        for start, end, ent in spans:
            token_idxs = [
                t for t, (o_s, o_e) in enumerate(offsets)
                if o_s < end and o_e > start
            ]
            if token_idxs:
                labels[token_idxs[0]] = entity_types.index(f"B-{ent}")
                for t in token_idxs[1:]:
                    labels[t] = entity_types.index(f"I-{ent}")

        all_labels.append(labels)

    # d) attach labels, drop offsets
    encodings["labels"] = all_labels
    encodings.pop("offset_mapping")
    return encodings

# 6) Apply in batched mode, dropping the old columns
tokenized = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=500,
    remove_columns=["output", "cleaned_output"]
)

# 7) Format for PyTorch & set up dynamic padding
tokenized.set_format("torch")
data_collator = DataCollatorForTokenClassification(tokenizer)

# 8) Model + Trainer
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)
training_args = TrainingArguments(
    output_dir="./ner_output",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    fp16=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 9) Train + save
trainer.train()
trainer.save_model("./fine_tuned_ner_model")
