# Improve pipeline: external speech-text data
# Train ASR on external transcribed speech and then use the updated ASR in pipeline 

data_size=$1
. scripts/asr/ft_e2e_asr.sh manifest/asr/ train_${data_size}_unlabeled

. scripts/pipeline/eval.sh train_${data_size}_unlabeled deberta_base dev combined t3
