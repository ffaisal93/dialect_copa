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
from utility import *




def get_all_fs_general(dataset, count=4,seed=42):

    fs_examps=[]
    for q in ['cause','effect']:
        data_q = dataset.filter(lambda example:  
                                                 example["question"]==q)
        for i in range(0,len(data_q),1):
            start=i
            end=i+count
            if end>len(data_q):
                borrow=end-len(data_q)
                indices=list(range(start,len(data_q)))+list(range(0,borrow))
            else:
                indices=list(range(start,end))
            fs_examps.append(dataset.select(indices))
    return fs_examps  

# def get_similar_sentences(corpus, queries,k=4):
#     top_k = min(k, len(corpus))
#     query_embeddings = sent_model.encode(queries, convert_to_tensor=True).to("cuda")
#     corpus_embeddings = sent_model.encode(corpus, convert_to_tensor=True).to("cuda")
#     corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
#     query_embeddings = util.normalize_embeddings(query_embeddings)
#     hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k,
#                                 score_function=util.dot_score)
#     all_hits={}
#     for i,hit in enumerate(hits):
#         all_hits[i]=[]
#         for j in range(0,top_k,1):
#             all_hits[i].append(hits[i][j]['corpus_id'])
#     return all_hits

# def get_all_fs_similar(train_data):
#     fs_examps=[]
#     count=0
#     for q in ['cause','effect']:
#         q_examples_corpus = train_data.filter(lambda example:  
#                                                  example["question"]==q)
#         q_examples_query = train_data.filter(lambda example:  
#                                                  example["question"]==q)
#         print(len(q_examples_query))
#         corpus=q_examples_corpus['premise']
#         query=q_examples_query['premise']
#         x=get_similar_sentences(corpus,query)
#         for i,indices in x.items():
#             fs_examps.append(q_examples_corpus.select(indices))
#     return fs_examps


def construct_prompt_general(ds_examples, lang='english'):
    
    question=ds_examples[0]['question']

    preamble = f"""You are a helpful assistant whose goal is to generate a premise, correct {question} and wrong {question}."""

    prompt_template_label0="""Instruction: 
    "premise": {premise}
    "correct {question}": {choice1}
    "wrong {question}": {choice2}"""

    prompt_template_label1="""Instruction:
    "premise": {premise}
    "correct {question}": {choice2}
    "wrong {question}": {choice1}"""

    
    prompt_examples = "\n".join([ prompt_template_label0.format(**d) 
                                   if d["label"]==0 
                                   else prompt_template_label1.format(**d)
                                   for d in ds_examples])
    prompt_examples=preamble+"\n\n\n"+prompt_examples

    prompt_to_gen=f"""\nInstruction: "premise": """
    return prompt_examples+prompt_to_gen

if __name__ == "__main__":
    dataset=load_datasets("copa-en")
    train_data=dataset['train']
    fs_examps_general=get_all_fs_general(train_data)
    fs_prompts_general=[construct_prompt_general(x) for x in fs_examps_general]
    print(fs_prompts_general[0])
    
    # from sentence_transformers import SentenceTransformer, util
    # sent_model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
    # fs_examps_similar=get_all_fs_similar(train_data)