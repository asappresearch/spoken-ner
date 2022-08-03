model_dir=$1
eval_set=$2

eval_label="raw"
model_type="deberta-base"
train_label="raw"

python spoken_ner/text_ner/ner_tagger.py  \
--data_dir manifest/text_ner \
--model_dir $model_dir \
--model_type $model_type \
--eval_asr False \
--train_label $train_label \
--eval_label $eval_label \
--eval_subset $eval_set 

