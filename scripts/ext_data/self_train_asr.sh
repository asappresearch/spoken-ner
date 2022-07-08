# Improve pipeline: external unlabeled speech
# Improve ASR module by training on psedolabeled speech to text data tagged using baseline ASR
# and then use the updated ASR in pipeline

data_size=$1
. scripts/asr/ft_e2e_asr.sh manifest/asr/ train_${data_size}_plabeled_asr

. scripts/pipeline/eval.sh train_${data_size}_plabeled_asr deberta_base dev combined t3