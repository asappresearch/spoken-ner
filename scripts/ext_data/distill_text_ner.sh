# Improve e2e: external speech-text data
# Train ASR on external transcribed speech and then use the updated ASR in pipeline 

data_size=$1

subset_name=train_${data_size}_plabeled_text_ner

. scripts/asr/ft_e2e_ner.sh manifest/e2e_ner/ $subset_name

. scripts/e2e_ner/eval_e2e_ner.sh $subset_name dev combined manifest/e2e_ner $subset_name