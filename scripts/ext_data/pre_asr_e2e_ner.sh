# Improve pipeline: external speech-text data
# Step 1: . pre_asr.sh
# Step 2: Finetune the pre-trained ASR on NER annotated data 

data_size=$1
. scripts/asr/ft_e2e_ner.sh manifest/e2e_ner/ fine-tune save/asr/w2v2_base_train_${data_size}_unlabeled

. scripts/e2e_ner/eval_e2e_ner.sh w2v2_pre_asr_fine-tune dev combined manifest/e2e_ner fine-tune