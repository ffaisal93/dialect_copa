from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MaLA-LM/mala-500", 
    local_dir="/scratch/ffaisal/dialect-copa/models/mala-500",
    local_dir_use_symlinks=False,
    token="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"
)


# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(
#     "/projects/antonis/fahim/models/Mistral-7B-v0.1",  
# )
# print(model)

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     "/scratch/ffaisal/catch/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720/", 
# )



# kernel install
# /scratch/ffaisal/DialectBench/vnv/vnv_trans_latest/bin/python -m ipykernel install --user --name 'transl'
