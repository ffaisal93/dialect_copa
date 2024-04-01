# DialectCOPA Data and Code Repository

This repository contains data and code for the paper "Data-Augmentation based Dialectal Adaptation for LLMs": GMUNLP's submission to the [Dialect-Copa](https://sites.google.com/view/vardial-2024/shared-tasks/dialect-copa?authuser=0) shared task organized by the [VARDIAL 2024 Workshop](https://sites.google.com/view/vardial-2024/home).

## Clone Repository
To get started, clone this repository to your local machine using the following command:

```
git clone https://github.com/ffaisal93/dialect_copa.git
```

## Installation
To install the necessary dependencies, run the following command:

```bash
./install.sh --task vnv_copa
```

This script will set up the required packages and libraries for the vnv_copa task.

## Experiments
### Activate Virtual Environment
Before running any experiments, activate the virtual environment by navigating to the `dialect_copa` directory and sourcing the `vnv_copa` environment:

```bash
cd dialect_copa
source vnv/vnv_copa
```

### Dataset Combinations
1. Preprocess the training data to create various data combinations by running the following commands:
    ```bash
    cd scripts
    python scripts/dataset_preprocess.py
    cd ..
    ```

   This operation will create the data settings as shown in the table below. The "Original names reported in Paper" column corresponds to the names used in Table 4 of the paper, while the "In data directory" column represents the corresponding dataset names in the `data` directory.

   | Original names reported in Paper | In data directory |
   |-----------------------------------|------------------|
   | o                                 | all_train        |
   | otrsl                             | orgl             |
   | otrslc                            | orglc            |
   | otrsl_{mk-hr-ckm}                 | ormg_mk_hr_ckm   |
   | otrsl_{hr-ckm}                    | orgl_hr_ckm      |
   | otrsl_{sl-cer}                    | orgl_sl_cer      |
   | otrsl_{sr-tor}                    | orgl_sr_tor      |
   | otrsl_{mix}                       | orgl_omix        |
   | otrslc_{mix}                      | orglc_omix       |
   | otrsl_{mix-mk-hr-ckm}             | orgl_omix_mk_hr_ckm |
   | otrsl_{mix-hr-ckm}                | orgl_omix_hr_ckm |
   | otrsl_{mix-sr-tor}                | orgl_omix_sr_tor |
   | otrsl_{mix-sl-cer}                | orgl_omix_sl_cer |
   | validation dataset                | all_val          |
2. Additional python code to create the reverse-augmented data (already performed)
   ```python
   import pandas
   import json
   import os
   
   def get_reverse(x):
    label_dict={0:'choice1',1:'choice2'}
    premise=x[label_dict[x['label']]]
    x[label_dict[x['label']]]=x['premise']
    x['premise']=premise
    if x['question']=='cause':
        x['question']='effect'
    else:
        x['question']='cause'
    return x

    DATADIR="../data"
    for f in os.listdir(DATADIR):
            if str(f).startswith('copa') and str(f).endswith("hr-ckm"):
                for f1 in os.listdir(os.path.join(DATADIR,f)):
                    if str(f1).startswith('train.jsonl'):
                        print(f,f1)
                        with open(os.path.join(DATADIR,f,f1), encoding="utf-8") as f2:
                            lines = f2.read().splitlines()
                        line_dicts = [json.loads(line) for line in lines]
                        print(os.listdir(os.path.join(DATADIR,f)))
                        reverse_line_dicts=[get_reverse(line.copy()) for line in line_dicts]
                        dest_file=os.path.join(DATADIR,f,'train-reverse.jsonl' )
                        output_file = open(dest_file, 'w',encoding='utf-8')
                        for dic in reverse_line_dicts:
                            json.dump(dic, output_file,ensure_ascii=False) 
                            output_file.write("\n")
                        output_file.close()
   ```

   

### Fine-tuning and Evaluation on bcms-bertic
1. We perform fine-tuning on the `bcms-bertic` model.
2. We evaluate the fine-tuned model using the Dialect-Copa `validation data` located in `data/all_val`.

   To run the fine-tuning and evaluation, execute the following command:

   ```bash
   ./install.sh --task train_predict_all_copa_encoder
   ```

3. If you have access to the test data and want to use it instead of the validation data, follow these steps:
   - First, prepare the test data dataset-dict using the provided Python code snippet.
   - Then, change the `TEST_FILE` variable in the `install.sh` file. Locate the line `TEST_FILE="data/all_val"` within the `if [[ "$task" = "train_predict_all_copa_encoder" || "$task" = "all" ]]; then` block and replace it with `TEST_FILE="data/dialect-copa-test/all_test"`.

### LORA Fine-tuning on Aya-101
To perform LORA fine-tuning on the Aya-101 model, run the following command:

```bash
./install.sh --task train_all_copa_aya_test
```

### Evaluate a LORA-tuned Aya on Validation Data
1. Modify line 25 in the `test_aya_copa.py` script to specify the location of the LORA-tuned model:

   ```python
   peft_model_id = "output_models/lora/orgl_mk_hr_ckm_test"
   ```

2. Run the evaluation script:

   ```bash
   mkdir results
   python scripts/test_aya_copa.py
   ```

### Evaluate All Test and Validation Data
To evaluate all available test and validation data using the base Aya-101 model, LORA-tuned models, and fine-tuned Bertic models, run the following command:

```bash
./install.sh --task run_copa_test_all
```

Note: You need to have access to the test data and create the `all_test` file beforehand.

Refer to the `if [[ "$task" = "run_copa_test_all" || "$task" = "all" ]]; then` block in the `install.sh` file for the complete command.

## Citation
If you use the data or code from this repository in your publication or study, please cite the following dataset and code papers:

```bibtex
@inproceedings{faisal-anastasopoulos-2024-augmentation,
    title = "Data-Augmentation based Dialectal Adaptation for LLMs",
    author = {Faisal, Fahim and Anastasopoulos, Antonios},
    booktitle = "Eleventh Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2024)",
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
}
```

```bibtex
@inproceedings{chifu-etal-2024-vardial,
    title = "Var{D}ial Evaluation Campaign 2024: Commonsense Reasoning in Dialects and Multi-Label Similar Language Identification",
    author = {Chifu, Adrian and Glava\v{s}, Goran and Ionescu, Radu and Ljube\v{s}i\'{c}, Nikola and Mileti\'{c}, Aleksandra and Mileti\'{c}, Filip and Scherrer, Yves and Vuli\'{c}, Ivan},
    editor = {Scherrer, Yves and Jauhiainen, Tommi and Ljube{\v{s}}i{\'c}, Nikola and Nakov, Preslav and Tiedemann, J{\"o}rg and Zampieri, Marcos},
    booktitle = "Eleventh Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2024)",
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
}
```

```bibtex
@inproceedings{ljubesic-etal-2024-dialect,
    title = "{DIALECT-COPA}: Extending the Standard Translations of the {COPA} Causal Commonsense Reasoning Dataset to South Slavic Dialects",
    author = {Ljube\v{s}i\'{c}, Nikola and Galant, Nada and Ben\v{c}ina, Sonja and \v{C}ibej, Jaka and Milosavljevi\'{c}, Stefan and Rupnik, Peter and Kuzman, Taja},
    editor = {Scherrer, Yves and Jauhiainen, Tommi and Ljube{\v{s}}i{\'c}, Nikola and Nakov, Preslav and Tiedemann, J{\"o}rg and Zampieri, Marcos},
    booktitle = "Eleventh Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2024)", 
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
}
```

