asr_model_type=$1
ner_model_dir=$2
eval_set=$3
eval_label=$4
lm=$5

# Decode audio to text using trained ASR
model_ckpt=`realpath save/asr/${asr_model_type}`
python -m slue_toolkit.eval.eval_w2v eval_ctc_model \
--model $model_ckpt \
--data manifest/slue-voxpopuli \
--subset ${eval_set} \
--lm ${lm}

# Post-process the decoded text to be compatible with transformers library
python -m slue_toolkit.text_ner.reformat_pipeline prep_data \
--model_type ${asr_model_type} \
--asr_data_dir manifest/slue-voxpopuli \
--asr_model_dir save/asr/${asr_model_type} \
--out_data_dir manifest/slue-voxpopuli/text_ner \
--eval_set $eval_set \
--lm $lm

eval_label="raw"
model_type="deberta-base"
train_label="raw"

# Run inference using trained NER model
python spoken_ner/text_ner/ner_tagger.py  \
--data_dir manifest/text_ner \
--model_dir $ner_model_dir \
--model_type $model_type \
--eval_asr False \
--train_label $train_label \
--eval_label $eval_label \
--eval_subset $eval_set \
--asr_model_type ${asr_model_type} \
--lm $lm
