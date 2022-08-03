# Improve pipeline: external speech-text data
# Step 1: . pre_asr.sh
# Step 2: Finetune the pre-trained ASR on NER annotated data 

data_size=$1
# Make sure you have run scripts/pre_asr.sh that saves ASR trained on $data_size data

# Resave ASR module after stripping away the classification head
python spoken_ner/prepare/resave_model_ckpt.py \
--ft_model_ckpt_dir save/asr/w2v2_base_train_${data_size}_unlabeled \
--pt_model_ckpt save/pretrained/w2v2_small.pt

# Finetune E2E NER starting from pre-trained ASR module
. scripts/asr/ft_e2e_ner.sh manifest/e2e_ner/ fine-tune save/asr/w2v2_base_train_${data_size}_unlabeled_no_proj

# Evaluate
. scripts/e2e_ner/eval.sh w2v2_base_pre_asr_${data_size}_fine-tune dev combined manifest vp_ner_fine-tune/4