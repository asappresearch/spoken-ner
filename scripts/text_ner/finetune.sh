manifest_dir=$1
train_subset=$2

label_type=raw
model_type=deberta-base

if [[ $train_subset == *"fine-tune"* ]]; then
    cfg_file=configs/deberta-base_fine-tune.json
elif [[ $train_subset == *"100h"* ]]; then
    cfg_file=configs/deberta-base_100h.json
elif [[ $train_subset == *"100h"* ]]; then
    cfg_file=configs/deberta-base_500h.json
fi

python -m slue_toolkit.text_ner.ner_deberta train \
--data_dir $manifest_dir/text_ner \
--model_dir save/text_ner/${model_type}_${label_type}_${train_subset} \
--model_type $model_type \
--label_type $label_type \
--train_subset $train_subset \
--cfg_file $cfg_file