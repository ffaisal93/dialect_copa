import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import random
import pandas as pd
import os



# Check whether the specified path exists or not
def make_dir(path="results"):
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    #print(f"wrote {file_path}")
    
def write_csv(filename,questions):
    df_format=pd.DataFrame.from_dict(questions)
    df_format.to_csv(filename,index=False)

def parse_choice(response,choices):
    multichar_choice=False
    for x in choices:
        if len(x)>1:
            multichar_choice=True
            break
    if len(response)==0:
        return None
    if multichar_choice==False:
        if len(response)==1:
            return choices.index(response[0]) + 1 if response[0] in choices else None
        elif response[0] in choices and not response[1].isalpha():
            return choices.index(response[0]) + 1
        else:
            return None
    elif multichar_choice==True:
        choice_found=None
        for choice in choices:
            if str(choice) in response:
                # print(choice,response,choices.index(choice) + 1)
                choice_found= choices.index(choice) + 1
        # if response in choices:
        #     return choices.index(response) + 1
        return choice_found
    
def dataset_randomize(ds):
    def modify(example):
        options = ['1', '2', '3', '4']
        random.Random(random.randrange(1,10)).shuffle(options)
        option_dict={str(i+1):x for i,x in enumerate(options)}
        option_dict_inv={x:str(i+1) for i,x in enumerate(options)}
        answers=[example['mc_answer1'],example['mc_answer2'],example['mc_answer3'],example['mc_answer4']]
        for k,v in option_dict.items():
            example['mc_answer'+k]=answers[int(v)-1]
        example['correct_answer_num']=option_dict_inv[example['correct_answer_num']]
        return example
    updated_dataset = ds.map(modify,keep_in_memory=True)
    return updated_dataset

def get_dataset(ds_conf,prompt_template,choices,example_count,leave_example,choice="original"):
    ds = load_dataset(**ds_conf)
    if choice=="random":
        ds = dataset_randomize(ds)
    ds_examples=ds.select(range(0,example_count))
    ds_prompts=ds.select(range(example_count,len(ds)-leave_example))

    prompt_examples = "\n\n".join([ prompt_template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1]) for d in ds_examples])
    prompts=[(prompt_examples + "\n\n" + prompt_template.format(**d, correct_answer="")).strip() for d in ds_prompts]
    return ds, ds_examples, ds_prompts, prompt_examples, prompts


def get_model(model_path,load_4bit=False,load_8bit=False):
    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        quantization_config=nf4_config if load_4bit else None,
        load_in_8bit=load_8bit,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer