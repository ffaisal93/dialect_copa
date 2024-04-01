#!/bin/bash
task=${task:-none}
lang=${lang:-eng}
lang2=${lang2:-eng}
lang3=${lang3:-eng}
MODEL_NAME=${MODEL_NAME:-bert}
CACHE_DIR=${CACHE_DIR:-'/scratch/ffaisal/hug_cache/datasets/DialectBench'}
dataset=${dataset:-wikiann}
prefix_text=${prefix_text:-prefix_text}

# RESULT_FOLDER="/projects/antonis/fahim/DialectBench/experiments"
ROOT_DIR="/scratch/ffaisal/dialect_copa"
RESULT_FOLDER="/scratch/ffaisal/dialect_copa/results"
TEST_RESULT_FOLDER="/scratch/ffaisal/dialect_copa/test/results"

mkdir ${RESULT_FOLDER}

while [ $# -gt 0 ]; do

  if [[ $1 == *"--"* ]]; then
    param="${1/--/}"
    declare $param="$2"
    echo $1 $2 #Optional to see the parameter:value result
  fi

  shift
done

if [[ "$MODEL_NAME" = "bert" ]]; then
  MODEL_PATH='bert-base-multilingual-cased'
fi

if [[ "$MODEL_NAME" = "xlmr" ]]; then
  MODEL_PATH='xlm-roberta-base'
fi

if [[ "$MODEL_NAME" = "xlmrl" ]]; then
  MODEL_PATH='xlm-roberta-large'
fi

echo ${task}
echo ${lang}
echo ${MODEL_NAME}
echo ${CACHE_DIR}

cd ${ROOT_DIR}
module load git
module load openjdk/11.0.2-qg

if [[ "$task" = "install_adapter" || "$task" = "all" ]]; then
  echo "------------------------------Install adapter latest------------------------------"
  module load python/3.8.6-ff
  rm -rf adapter-transformers-l
  rm -rf vnv/vnv-adp-l
  python -m venv vnv/vnv-adp-l
  source vnv/vnv-adp-l/bin/activate
  wget -O adapters3.1.0.tar.gz https://github.com/adapter-hub/adapter-transformers/archive/refs/tags/adapters3.1.0.tar.gz
  tar -xf adapters3.1.0.tar.gz
  rm adapters3.1.0.tar.gz
  mv adapter-transformers-adapters3.1.0 adapter-transformers-l
  cd adapter-transformers-l
  #cp ../scripts/ad_l_trans_trainer.py src/transformers/trainer.py
  pip install .
  ../vnv/vnv-adp-l/bin/python -m pip install --upgrade pip
  cd ..
  pip install --upgrade pip
  pip3 install -r requirements.txt
  ##for A100 gpu
  deactivate
fi

if [[ "$task" = "install_copa" || "$task" = "all" ]]; then
  module load python/3.8.6-ff
  module load git
  rm -rf vnv/vnv_copa
  python -m venv vnv/vnv_copa
  source vnv/vnv_copa/bin/activate
  pip install --upgrade pip
  pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
  pip install ipykernel
  pip install transformers==4.34.0
  ##pip install transformers==4.28.0##original
  pip install sentencepiece
  pip install peft
  pip install git+https://github.com/huggingface/datasets
  pip install git+https://github.com/huggingface/huggingface_hub
  pip install bitsandbytes==0.40.2 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
  pip install bitsandbytes
  pip install "accelerate>=0.20.3" "evaluate" tensorboard scikit-learn
  /scratch/ffaisal/dialect-copa/vnv/vnv_copa/bin/python -m ipykernel install --user --name 'copa'
  deactivate

fi

if [[ "$task" = "install_translation" || "$task" = "all" ]]; then
  module load python/3.8.6-ff
  module load git
  rm -rf vnv/vnv_translation
  python -m venv vnv/vnv_translation
  source vnv/vnv_translation/bin/activate
  pip install --upgrade pip
  pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
  pip install transformers==4.34.0
  pip install ctranslate2 >=3.16.0
  cd ${ROOT_DIR}
  deactivate

fi

if [[ "$task" = "vnv_copa" || "$task" = "all" ]]; then
  module load git
  module load cuda/12.2.0.1
  module load gnu10/10.3.0-ya
  module load python/3.9.9-jh
  python -m venv vnv/vnv_copa
  source vnv/vnv_copa/bin/activate
  pip install --upgrade pip
  pip install transformers==4.31.0
  pip install peft==0.4.0
  pip install accelerate==0.21.0
  # pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
  pip install bitsandbytes==0.40.2
  pip install fsspec==2023.6.0
  pip install datasets==2.10.1
  pip install trl==0.7.11
  pip install scipy
  pip install ipykernel
  python -m ipykernel install --user --name vnv_copa --display-name "vnv_copa"
  cd ${ROOT_DIR}
  deactivate

fi


if [[ "$task" = "test" || "$task" = "all" ]]; then
  module load git
  module load cuda/12.2.0.1
  module load gnu10/10.3.0-ya
  module load python/3.9.9-jh
  python -m venv vnv/test
  source vnv/test/bin/activate
  pip install --upgrade pip
  pip install transformers==4.31.0
  pip install peft==0.4.0
  pip install accelerate==0.21.0
  # pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
  pip install bitsandbytes==0.40.2
  pip install fsspec==2023.6.0
  pip install datasets==2.10.1
  pip install trl==0.7.11
  pip install scipy
  # pip install --upgrade pip
  # pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
  # pip install transformers==4.34.0
  # pip install ctranslate2>=3.16.0
  cd ${ROOT_DIR}
  deactivate

fi

if [[ "$task" = "test_jupyter" || "$task" = "all" ]]; then
  module load git
  module load cuda/12.2.0.1
  module load gnu10/10.3.0-ya
  module load python/3.9.9-jh
  python -m venv vnv/test_jupyter
  source vnv/test_jupyter/bin/activate
  pip install --upgrade pip
  pip install transformers==4.31.0
  pip install peft==0.4.0
  pip install accelerate==0.23.0
  # pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
  pip install bitsandbytes==0.40.2
  pip install fsspec==2023.6.0
  pip install datasets==2.10.1
  pip install scipy
  # pip install --upgrade pip
  # pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
  # pip install transformers==4.34.0
  # pip install ctranslate2>=3.16.0
  cd ${ROOT_DIR}
  deactivate

fi



if [[ "$task" = "translate_xcopa" || "$task" = "all" ]]; then
  ##lang codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
  source vnv/vnv_translation/bin/activate
  # rm -rf tempdata
  # mkdir tempdata
  # python scripts/translation_nli/save_data.py
  dataset="data/genx_xcopa/test.txt"
  lang="srp_Cyrl"
  start=$(date +%s)
  echo "${dataset}"
  python Easy-Translate/translate.py \
    --sentences_path ${dataset} \
    --output_path result/nllbb-${lang}.txt \
    --source_lang eng_Latn \
    --target_lang ${lang} \
    --model_name /projects/antonis/fahim/models/nllb-200-3.3B \
    --precision 8 \
    --starting_batch_size 128

  end=$(date +%s)
  min=60
  runtime=$((end - start))
  echo "runtime----------" $((runtime / min))

  deactivate

fi

if [[ "$task" = "train_copa_encoder" || "$task" = "all" ]]; then

  source vnv/vnv_copa/bin/activate
  TASK="xcopa"
  MODEL="classla/bcms-bertic"
  SAVE_DIR="output_models"
  GRAD_ACC=1
  LR=2e-5
  EPOCH=5
  MAXL=128
  BATCH_SIZE=16
  TRAIN_FILE="data/all_train_rev"
  VALIDATION_FILE="data/all_val"
  python scripts/run_copa.py \
    --task ${TASK} \
    --model_name_or_path $MODEL \
    --output_dir $SAVE_DIR/ \
    --train_data $TRAIN_FILE \
    --validation_data $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --max_seq_length $MAXL \
    --num_train_epochs $EPOCH
  deactivate
fi

if [[ "$task" = "train_predict_all_copa_encoder" || "$task" = "all" ]]; then

  source vnv/vnv_copa/bin/activate
  TASK="xcopa"
  MODEL="classla/bcms-bertic"
  GRAD_ACC=1
  LR=2e-5
  EPOCH=5
  MAXL=128
  BATCH_SIZE=64
  VALIDATION_FILE="data/all_val"
  TEST_FILE="data/all_val"

  export ALL_DATAFILE=("orglc" "orgl" "orglc_omix" "orgl_omix" "orglc_sr_tor" "orglc_mk_hr_ckm" "orglc_omix_sr_tor" "orglc_omix_mk_hr_ckm" "orgl_sr_tor" "orgl_mk_hr_ckm" "orgl_sl_cer" "orgl_hr_ckm" "orgl_omix_sr_tor" "orgl_omix_mk_hr_ckm" "orgl_omix_sl_cer" "orgl_omix_hr_ckm")


  # export ALL_MODELS=("UD_English-EWT")
  # export ALL_DATAFILE=("all_train")
  for TRAIN_DATA1 in "${ALL_DATAFILE[@]}"; do
    echo "${TRAIN_DATA1}"

    TRAIN_DATA="${TRAIN_DATA1##*:}"

    VAL_LANG="${TRAIN_DATA1%%:*}"

    echo ${VAL_LANG}
    echo ${TRAIN_DATA}
    SAVE_DIR="output_models/${TRAIN_DATA}"
    TRAIN_FILE="data/${TRAIN_DATA}"
    VAL_RESULT_FILE="results/val_${TRAIN_DATA}_10.txt"
    TEST_RESULT_FILE="results/test_${TRAIN_DATA}_10.txt"
    MODEL=$SAVE_DIR
    rm "${VAL_RESULT_FILE}"
    rm "${TEST_RESULT_FILE}"
    python scripts/run_copa.py \
      --task ${TASK} \
      --model_name_or_path $MODEL \
      --val_lang ${VAL_LANG} \
      --output_dir "${SAVE_DIR}_10" \
      --train_data $TRAIN_FILE \
      --validation_data $VALIDATION_FILE \
      --test_data $TEST_FILE \
      --val_result_file $VAL_RESULT_FILE \
      --test_result_file $TEST_RESULT_FILE \
      --save_steps 100 \
      --do_train \
      --do_eval \
      --eval_steps 100 \
      --evaluation_strategy steps \
      --do_predict \
      --overwrite_output_dir \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --learning_rate $LR \
      --max_seq_length $MAXL \
      --save_total_limit 5 \
      --load_best_model_at_end True \
      --num_train_epochs $EPOCH \
      --metric_for_best_model eval_accuracy
    rm -rf $SAVE_DIR/checkpoint*
  done
  deactivate
fi

if [[ "$task" = "train_predict_all_copa_encoder_select" || "$task" = "all" ]]; then

  source vnv/vnv_copa/bin/activate
  TASK="xcopa"
  MODEL="classla/bcms-bertic"
  GRAD_ACC=1
  LR=2e-5
  EPOCH=10
  MAXL=128
  BATCH_SIZE=16
  VALIDATION_FILE="data/all_val"
  TEST_FILE="data/dialect-copa-test/all_test"

  export ALL_DATAFILE=("all_train_select" "all_train_rev_select" "all_train_rev_genx_select" "all_train_rev_genx_omixmatch_select" "all_train_rev_genx_wmixmatch_select")

  # export ALL_MODELS=("UD_English-EWT")
  # export ALL_DATAFILE=("all_train")
  for TRAIN_DATA in "${ALL_DATAFILE[@]}"; do
    echo "${TRAIN_DATA}"

    SAVE_DIR="output_models/${TRAIN_DATA}"
    TRAIN_FILE="data/${TRAIN_DATA}"
    VAL_RESULT_FILE="results/val_${TRAIN_DATA}.txt"
    TEST_RESULT_FILE="results/test_${TRAIN_DATA}.txt"
    rm "${VAL_RESULT_FILE}"
    rm "${TEST_RESULT_FILE}"
    python scripts/run_copa.py \
      --task ${TASK} \
      --model_name_or_path $MODEL \
      --output_dir $SAVE_DIR/ \
      --train_data $TRAIN_FILE \
      --validation_data $VALIDATION_FILE \
      --test_data $TEST_FILE \
      --val_result_file $VAL_RESULT_FILE \
      --test_result_file $TEST_RESULT_FILE \
      --save_steps 4000 \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --learning_rate $LR \
      --max_seq_length $MAXL \
      --num_train_epochs $EPOCH
  done
  deactivate
fi

if [[ "$task" = "train_predict_all_copa_encoder_test" || "$task" = "all" ]]; then

  source vnv/test/bin/activate
  TASK="xcopa"
  MODEL="classla/bcms-bertic"
  GRAD_ACC=1
  LR=2e-5
  EPOCH=10
  MAXL=128
  BATCH_SIZE=16
  VALIDATION_FILE="data/all_val"
  TEST_FILE="data/dialect-copa-test/all_test"

  export ALL_DATAFILE=("all_train_select")

  # export ALL_MODELS=("UD_English-EWT")
  # export ALL_DATAFILE=("all_train")
  for TRAIN_DATA in "${ALL_DATAFILE[@]}"; do
    echo "${TRAIN_DATA}"

    SAVE_DIR="output_models/${TRAIN_DATA}_test"
    TRAIN_FILE="data/${TRAIN_DATA}"
    VAL_RESULT_FILE="results/val_${TRAIN_DATA}_test.txt"
    TEST_RESULT_FILE="results/test_${TRAIN_DATA}_test.txt"
    rm "${VAL_RESULT_FILE}"
    rm "${TEST_RESULT_FILE}"
    python scripts/run_copa.py \
      --task ${TASK} \
      --model_name_or_path $MODEL \
      --output_dir $SAVE_DIR/ \
      --train_data $TRAIN_FILE \
      --validation_data $VALIDATION_FILE \
      --test_data $TEST_FILE \
      --val_result_file $VAL_RESULT_FILE \
      --test_result_file $TEST_RESULT_FILE \
      --save_steps 4000 \
      --do_train \
      --do_eval \
      --do_predict \
      --overwrite_output_dir \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --learning_rate $LR \
      --max_seq_length $MAXL \
      --num_train_epochs $EPOCH \
      --max_steps 10
  done
  deactivate
fi

if [[ "$task" = "train_all_copa_aya_test" || "$task" = "all" ]]; then

  source vnv/vnv_copa/bin/activate
  TASK="xcopa"
  MODEL="models/aya-101"
  # MODEL="google-t5/t5-small"
  GRAD_ACC=4
  LR=2e-5
  EPOCH=5
  MAXL=128
  BATCH_SIZE=16
  VALIDATION_FILE="data/all_val"
  TEST_FILE="data/dialect-copa-test/all_test"

  export ALL_DATAFILE=("copa-sl-cer:orgl" "copa-sl-cer:orglc_omix_mk_hr_ckm" "copa-sl-cer:orgl_sl_cer" "copa-ck:orgl_mk_hr_ckm" "copa-sr-tor:orgl_hr_ckm" "copa-sr-tor:orglc_omix_sr_tor" "copa-sr-tor:orglc_omix_sr_tor" "copa-sr-tor:orglc_sr_tor")

  export ALL_DATAFILE=("orgl_hr_ckm")

  # export ALL_MODELS=("UD_English-EWT")
  # export ALL_DATAFILE=("all_train")
  for TRAIN_DATA in "${ALL_DATAFILE[@]}"; do
    echo "${TRAIN_DATA}"

    SAVE_DIR="output_models/lora/${TRAIN_DATA}_test"
    TRAIN_FILE="data/${TRAIN_DATA}"
    VAL_RESULT_FILE="results/val_${TRAIN_DATA}_test.txt"
    TEST_RESULT_FILE="results/test_${TRAIN_DATA}_test.txt"
    rm "${VAL_RESULT_FILE}"
    rm "${TEST_RESULT_FILE}"

    python scripts/run_copa_lora.py \
      --task ${TASK} \
      --model_name_or_path $MODEL \
      --output_dir $SAVE_DIR/ \
      --train_data $TRAIN_FILE \
      --validation_data $VALIDATION_FILE \
      --test_data $TEST_FILE \
      --val_result_file $VAL_RESULT_FILE \
      --test_result_file $TEST_RESULT_FILE \
      --save_steps 500 \
      --do_train \
      --overwrite_output_dir \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_device_train_batch_size $BATCH_SIZE \
      --learning_rate $LR \
      --max_seq_length $MAXL \
      --num_train_epochs $EPOCH
    # --max_steps 10
  done
  deactivate
fi

if [[ "$task" = "run_copa_test_all" || "$task" = "all" ]]; then

  source vnv/vnv_copa/bin/activate
  TASK="xcopa"
  MODEL="classla/bcms-bertic"
  # MODEL="google-t5/t5-small"
  GRAD_ACC=4
  LR=2e-5
  EPOCH=5
  MAXL=128
  BATCH_SIZE=16
  VALIDATION_FILE="data/all_val"
  TEST_FILE="data/dialect-copa-test/all_test"

  export ALL_DATAFILE=("orgl_hr_ckm")

  # export ALL_MODELS=("UD_English-EWT")
  # export ALL_DATAFILE=("all_train")
  for TRAIN_DATA in "${ALL_DATAFILE[@]}"; do
    echo "${TRAIN_DATA}"

    SAVE_DIR="output_models/lora/${TRAIN_DATA}_test"
    TRAIN_FILE="data/${TRAIN_DATA}"
    VAL_RESULT_FILE="results/val_${TRAIN_DATA}_test.txt"
    TEST_RESULT_FILE="results/test_${TRAIN_DATA}_test.txt"
    rm "${VAL_RESULT_FILE}"
    rm "${TEST_RESULT_FILE}"

    python scripts/run_copa_test.py \
      --task ${TASK} \
      --model_name_or_path $MODEL \
      --output_dir $SAVE_DIR/ \
      --train_data $TRAIN_FILE \
      --validation_data $VALIDATION_FILE \
      --test_data $TEST_FILE \
      --val_result_file $VAL_RESULT_FILE \
      --test_result_file $TEST_RESULT_FILE \
      --save_steps 500 \
      --do_predict_test_all \
      --overwrite_output_dir \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_device_train_batch_size $BATCH_SIZE \
      --learning_rate $LR \
      --max_seq_length $MAXL \
      --num_train_epochs $EPOCH
    # --max_steps 10
  done
  deactivate
fi



