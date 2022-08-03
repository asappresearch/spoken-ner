model_name=$1

python spoken_ner/analysis/error_analysis.py get_error_dist \
--model_dir save/text_ner/${model_name} 
