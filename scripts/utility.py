import argparse
import os
import csv
import json
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict
from transformers import logging
from generate_utility import *
from transformers.generation import GenerationConfig
from peft import PeftModel
# import bitsandbytes as bnb


lang="Croatian"
preamble = f"""You are a helpful following assistant whose goal is to select the preferred (least wrong) output for a given instruction in {lang}."""

prompt_template="""Instruction: Given the premise, ""{premise}"", What is the correct {question}?
{question} A: {choice1}
{question} B: {choice2}
Correct {question}: {correct_answer}"""

choices=["A","B"]

def get_few_shot_examples(dataset, fs_per_label=1, seed=42):
    labels = list(set(dataset["label"]))
    few_shot_examples = []
    for label in labels:
        label_examples = dataset.filter(lambda example: example["label"] == label and example["question"]=='cause')
        # shuffle the examples
        label_examples = label_examples.shuffle(seed=seed)
        # get the first fs_per_label examples
        label_examples = label_examples.select(
            range(min(fs_per_label, len(label_examples)))
        )
        few_shot_examples += [example for example in label_examples]
        
        label_examples = dataset.filter(lambda example: example["label"] == label and example["question"]=='effect')
        # shuffle the examples
        label_examples = label_examples.shuffle(seed=seed)
        # get the first fs_per_label examples
        label_examples = label_examples.select(
            range(min(fs_per_label, len(label_examples)))
        )
        few_shot_examples += [example for example in label_examples]

    # Shuffle the few shot examples
    random.shuffle(few_shot_examples)
    return few_shot_examples

def construct_prompt(ds_examples):
    def example_to_prompt(example, add_label=True):
        ex_prompt = f"Sentence: {example['text']}\n"
        if add_label:
            ex_prompt += f"Label: {example['label']}\n"
        return ex_prompt

    # To Do: Add domain of the text in the instruction like "In this task you given text from {domain}

    # Format the first five rows as examples for 5-shot prompting
    prompt_examples = "\n\n".join([ prompt_template.format(**d,correct_answer=choices[int(d["label"])-1]) for d in ds_examples])
    prompt_examples=preamble+"\n\n\n"+prompt_examples
    return prompt_examples


def load_datasets(lang,DATADIR='../data'):
    from pathlib import Path


    datasets = {}
    for split in ["train", "val"]:
        path=Path(DATADIR,lang,f"{split}.jsonl")
        if not path.exists():
            path=Path(DATADIR,lang,f"{split}.trans.jsonl")
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        line_dicts = [json.loads(line) for line in lines]
        datasets[split] = pd.DataFrame(line_dicts)
        datasets[split] = Dataset.from_pandas(datasets[split])
        # datasets[split] = datasets[split].filter(
        #     lambda example: not (
        #         example["text"] == "sentence" and example["label"] == "label"
        #     )
        # )
        print(f"{split} size: {len(datasets[split])}")

    datasets = DatasetDict(datasets)
    return datasets

    return datasets