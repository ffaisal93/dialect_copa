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
from utility import *
from transformers.generation import GenerationConfig
from peft import PeftModel



import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "output_models/lora/orgl_mk_hr_ckm_test"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,   device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})



# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# checkpoint = "../models/aya-101"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")

def generate_result(prompts,gen_config,model_name='aya',bs=8):
    all_response=[]
    all_response_raw=[]
    end=len(prompts)
    for start in tqdm(range(0,end,bs)):
        stop=min(start+bs,len(prompts))
        if start<stop:
            prompts_batch=prompts[start:stop]
            encodings=tokenizer(prompts_batch, return_tensors="pt", padding='longest', truncation=False).to("cuda")
            with torch.no_grad():
                output_ids = model.generate(**encodings, **gen_config)
            responses=tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for i,response_raw in enumerate(responses):
                sample_no=i+start
                if model_name!='aya':
                    response=response_raw[len(prompts[sample_no]):]
                    response=response.split("\n")[0].strip() if "\n" in response else response.strip()
                else:
                    response=response_raw[-1]
                all_response.append(response)
                all_response_raw.append(response_raw)
                
    return all_response_raw,all_response

def eval_result(all_preds,all_true_labels):
    count=0
    ind_true=[]
    not_true=[]
    indx=[]
    for i,res in enumerate(all_preds):
        if res in choices:
            if choices.index(res)==all_true_labels[i]:
                count+=1
                ind_true.append(i)
            else:
                not_true.append(i)
                indx.append(res)
    acc=count/len(all_preds)
    # print(acc, count, len(all_preds), len(all_true_labels), not_true,indx)
    return acc




def get_few_shot_examples(dataset, question,fs_per_label=2, seed=42):
    labels = list(set(dataset["label"]))
    few_shot_examples = []
    for label in labels:
        label_examples = dataset.filter(lambda example: example["label"] == label and example["question"]==question)
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

def test_construct_Prompt(ds_examples,min_ex=4):
    ds_examples=ds_examples[:min_ex]
    prompt_examples = "\n\n".join([ prompt_template_cause.format(**d,correct_answer=choices[int(d["label"])]) 
                                   if d["question"]=='cause' 
                                   else prompt_template_effect.format(**d,correct_answer=choices[int(d["label"])])
                                   for d in ds_examples])
    prompt_examples=preamble+"\n\n\n"+prompt_examples
    return prompt_examples

def construct_single(row,fs_prompt):
    if row['question']=='cause':
        prompt=(fs_prompt + "\n\n" + prompt_template_cause.format(**row, correct_answer="")).strip()
        # prompt=(prompt_template_cause.format(**row, correct_answer="")).strip()
    else:
        prompt=(fs_prompt + "\n\n" + prompt_template_effect.format(**row, correct_answer="")).strip()
        # prompt=( prompt_template_cause.format(**row, correct_answer="")).strip()
    return prompt


gen_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "do_sample": True,
                "max_new_tokens": 5,
                "pad_token_id": tokenizer.eos_token_id
                    }

tokenizer.pad_token_id = tokenizer.eos_token_id



all_lang_data={
    'english':'copa-en',
    'croatian':'copa-hr',
    'Slovenian':'copa-sl',
    'Cerkno-dialect-of-Slovenian':'copa-sl-cer',
    'Serbian':'copa-sr',
    'Torlak-dialect':'copa-sr-tor',
    'Macedonian':'copa-mk'  
}

all_results=[]
eval_type='val'
for lang,dfile in all_lang_data.items():

    choices=["A","B"]
    
    preamble = f"""You are a helpful assistant whose goal is to select the correct output for a given instruction in {lang}."""

    prompt_template_cause="""Instruction: Given the premise, ""{premise}"", What is the correct {question} before this?
    A: {choice1}
    B: {choice2}
    Correct {question}: {correct_answer}"""

    prompt_template_effect="""Instruction: Given the premise, ""{premise}"", What is the correct {question} after this?
    A: {choice1}
    B: {choice2}
    Correct {question}: {correct_answer}"""
    
    dataset=load_datasets(dfile,DATADIR='data')
    
    print(dataset['val'][0])
    
    all_val_prompts=[]
    all_val_labels=[]
    fs_examp_cause=get_few_shot_examples(dataset['train'],'cause',fs_per_label=2,seed=41)
    fs_examp_effect=get_few_shot_examples(dataset['train'],'effect',fs_per_label=2,seed=42)
    fs_prompt_cause=test_construct_Prompt(fs_examp_cause)
    fs_prompt_effect=test_construct_Prompt(fs_examp_effect)
    for row in dataset['val']:
        if row['question']=='effect':
            prompt=(fs_prompt_effect + "\n\n" + prompt_template_effect.format(**row, correct_answer="")).strip()
        else:
            prompt=(fs_prompt_cause + "\n\n" + prompt_template_cause.format(**row, correct_answer="")).strip()
        all_val_prompts.append(prompt)
        all_val_labels.append(row['label'])
        
    all_response_raw,all_response=generate_result(all_val_prompts,gen_config,'aya')
    
    acc=eval_result(all_response,all_val_labels)
    res={
        'name':f'{eval_type}-{dfile}',
        'test_accuracy': acc,
        'test_loss': 0,
        'test_runtime': 0,
        'test_samples_per_second': 0,
        'test_steps_per_second': 0
    }
    all_results.append(res)

    
    
output = open(f"results/{eval_type}_aya_orgl_hr_mk_ckm.txt", "w")
for k in all_results:
    for vk,vv in k.items():
        if vk=='name':
            output.writelines(f'{vv}\n')
        else:
            output.writelines(f'{vk} = {vv:.4f}\n')
    output.writelines(f'\n')    