# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning models for the XCOPA and SiQA multiple choice tasks.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
import time
import json

import numpy as np
import torch
from datasets import load_dataset, load_from_disk,DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint, is_main_process


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    train_data: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_data: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    test_data: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    task: Optional[str] = field(default='xcopa', metadata={"help": "The training task."})
    train_lang: Optional[str] = field(default='en', metadata={"help": "The language of the data."})
    val_lang: Optional[str] = field(default='copa-en', metadata={"help": "The language of the data."})
    val_result_file: Optional[str] = field(default='val_result.txt', metadata={"help": "The language of the data."})
    test_result_file: Optional[str] = field(default='test_result.txt', metadata={"help": "The language of the data."})
    predict_langs: Optional[str] = field(default=None, metadata={"help": "The language of the data."})

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) if label_name in feature else 0 for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task == 'xcopa':
        if data_args.train_data is not None:
            datasets=load_from_disk(data_args.train_data)
            datasets_val=load_from_disk(data_args.validation_data)
            datasets_test=load_from_disk(data_args.test_data)
            datasets['validation']=datasets_val[data_args.val_lang]
            print(datasets)
            print(datasets_val)
            print(datasets_test)
        else:
            datasets = load_dataset("xcopa_data", data_args.train_lang)
    elif data_args.task == 'siqa':
        datasets = load_dataset("siqa_data", data_args.train_lang)
    else:
      return ValueError(f'{task} is not recognized.')

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    # ending_names = [f"ending{i}" for i in range(4)]
    # context_name = "sent1"
    # question_header_name = "sent2"

    if data_args.task == 'xcopa':
      context_name = "premise"
      question_header_name = "question"
      ending_names = [f"choice{i}" for i in [1, 2]]
      num_answers = 2
    elif data_args.task == 'siqa':
      context_name = 'context'
      question_header_name = 'question'
      ending_names = [f"answer{i}" for i in ['A', 'B', 'C']]
      num_answers = 3

    # premise + question <SEP> choice1, premise + question <SEP> choice2

    # Preprocessing the datasets.
    def preprocess_function(examples):
        if data_args.task == 'xcopa':
            first_sentences = [
              [f"{context} What was the {question}?"] * num_answers for context, question in zip(examples[context_name], examples[question_header_name])
           ]
        elif data_args.task == 'siqa':
            first_sentences = [
              [f"{context} {question}"] * num_answers for context, question in zip(examples[context_name], examples[question_header_name])
           ]
        second_sentences = [
            [f"{examples[end][i]}" for end in ending_names] for i, _ in enumerate(first_sentences)
        ]

        # first_sentences = [[context] * 4 for context in examples[context_name]]
        # question_headers = examples[question_header_name]
        # second_sentences = [
        #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        # ]

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Un-flatten
        # unflattened = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        unflattened = {k: [v[i : i + num_answers] for i in range(0, len(v), num_answers)] for k, v in tokenized_examples.items()}
        return unflattened

    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        # if last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # elif os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        # else:
        #     checkpoint = None
        checkpoint = None  # We want to train on COPA at global step 0
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{data_args.train_lang}_{data_args.task}.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    if data_args.predict_langs is not None and training_args.do_predict:
      predict_langs = data_args.predict_langs.split(',')
      scores = []
      for lang in predict_langs:
        logger.info(f"*** Predicting on {lang} ***")

        datasets = load_dataset("xcopa_data", lang)
        tokenized_datasets = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        eval_dataset = tokenized_datasets["test"]

        results = trainer.predict(eval_dataset)
        predictions = np.argmax(results[0], 1)
        output_pred_file = os.path.join(training_args.output_dir, f"test-{lang}.jsonl")
        with open(output_pred_file, 'w') as f:
          for pred in predictions:
            json_str = json.dumps({'label': int(pred)})
            f.write(json_str + '\n')
    if data_args.predict_langs is None and training_args.do_predict:
      pred_data_type='val'
      predict_langs = [x for x in datasets_val]
      scores = []
      # predict_langs=['copa-en']
      for lang in predict_langs:
        logger.info(f"*** Predicting on {lang} ***")

        datasets = datasets_val[lang]
        tokenized_datasets = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        eval_dataset = tokenized_datasets

        results = trainer.predict(eval_dataset)
        print(results[2])
        predictions = np.argmax(results[0], 1)
        output_pred_file = os.path.join(training_args.output_dir, f"{pred_data_type}-{lang}.jsonl")
        # output_pred_file_r = os.path.join(training_args.output_dir, f"{data_args.pred_data_type}-{lang}.txt")
        output_result_file = os.path.join(data_args.val_result_file)
        with open(output_pred_file, 'w') as f:
          for pred in predictions:
            json_str = json.dumps({'label': int(pred)})
            f.write(json_str + '\n')
        with open(output_result_file, "a") as writer:
            logger.info("***** Pred results *****")
            logger.info(f"{pred_data_type}-{lang}")        
            writer.write(f"{pred_data_type}-{lang}\n")
            for key, value in sorted(results[2].items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
            writer.write(f"\n")
        

        
        
      pred_data_type='test'
      predict_langs = [x for x in datasets_test]
      scores = []
      # predict_langs=['copa-en']
      for lang in predict_langs:
        logger.info(f"*** Predicting on {lang} ***")

        datasets = datasets_test[lang]
        tokenized_datasets = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        eval_dataset = tokenized_datasets

        results = trainer.predict(eval_dataset)
        print(results[2])
        predictions = np.argmax(results[0], 1)
        output_pred_file = os.path.join(training_args.output_dir, f"{pred_data_type}-{lang}.jsonl")
        output_result_file = os.path.join(data_args.test_result_file)
        with open(output_pred_file, 'w') as f:
          for pred in predictions:
            json_str = json.dumps({'label': int(pred)})
            f.write(json_str + '\n')
        # with open(output_result_file, "a") as writer:
        #     logger.info("***** Pred results *****")
        #     logger.info(f"{pred_data_type}-{lang}")        
        #     writer.write(f"{pred_data_type}-{lang}\n")
        #     for key, value in sorted(results[2].items()):
        #         logger.info(f"  {key} = {value}")
        #         writer.write(f"{key} = {value}\n")
        #     writer.write(f"\n")  

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()