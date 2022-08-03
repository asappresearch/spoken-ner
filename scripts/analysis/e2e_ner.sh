model_name=$1
lm=$2

python spoken_ner/analysis/error_analysis.py get_error_dist \
--model_dir save/e2e_ner/${model_name}/metrics/error_analysis \
--e2e_ner_lm $lm 
