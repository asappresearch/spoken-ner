ner_model_name=$1
asr_model_name=$2

python spoken_ner/analysis/error_analysis.py get_error_dist \
--model_dir save/text_ner/${ner_model_name}/ \
--asr_model_name $asr_model_name