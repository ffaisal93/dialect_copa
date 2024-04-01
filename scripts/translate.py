# Load the model and tokenizer
# Important: This should be done only once

import ctranslate2
import transformers

src_lang = "eng_Latn"
tgt_lang = "mkd_Cyrl"

device = "cuda"  # or "cpu"
beam_size = 4

translator = ctranslate2.Translator("/scratch/ffaisal/dialect-copa/models/ct2fast-nllb-200-3.3B", device)
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=src_lang)

src_file="data/genx_xcopa/eng.txt"
dest_file="data/genx_xcopa/mkd.txt"
with open(src_file) as f:
    source_sents = f.read().splitlines()

source_sents=source_sents
all_translations=[]
for i in range(0,len(source_sents),64):
    print(i,round((i*100)/len(source_sents)))
    source_sents_batch=source_sents[i:min(i+64,len(source_sents))]
    source_sents_tokenized = tokenizer(source_sents_batch)
    #print(source_sents_tokenized)
    source = [tokenizer.convert_ids_to_tokens(sent) for sent in source_sents_tokenized["input_ids"]]
    target_prefix = [[tgt_lang]] * len(source_sents)
    results = translator.translate_batch(source, target_prefix=target_prefix, beam_size=beam_size)
    #print(results)
    target_sents_tokenized = [result.hypotheses[0][1:] for result in results]
    #print(target_sents_tokenized)

    target_sents_to_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in target_sents_tokenized]
    translations = tokenizer.batch_decode(target_sents_to_ids)

    # print("Translations:", *translations, sep="\n")
    all_translations.extend(translations)

with open(dest_file, 'w') as f:
    for line in all_translations:
        f.write(f"{line}\n")