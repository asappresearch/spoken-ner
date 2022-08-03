model_name=$1
subset=$2
manifest_dir=$3
lm=$4 # nolm or vp_ner/4

beam=500
lm_wt=2
ws=1

# This saves the decoded text at save/e2e_ner/${model_name}/decode
python -m  slue_toolkit.eval.eval_w2v eval_ctc_model \
--model save/e2e_ner/${model_name} \
--data ${manifest_dir}/e2e_ner \
--subset ${subset} \
--lm $lm \
--beam_size $beam \
--lm_weight $lm_wt \
--word_score $ws 
