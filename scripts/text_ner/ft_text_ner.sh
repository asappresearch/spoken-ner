manifest_dir=$1
train_subset=$2

label_type=raw
model_type=deberta-base

python -m slue_toolkit.text_ner.ner_deberta train \
--data_dir $manifest_dir \
--model_dir save/nlp_ner/${model_type}_${label_type}_${train_subset} \
--model_type $model_type \
--label_type $label_type \
--train_subset $train_subset 