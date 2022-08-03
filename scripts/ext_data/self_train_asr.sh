# Improve pipeline: external unlabeled speech
# Improve ASR module by training on psedolabeled speech to text data tagged using baseline ASR
# and then use the updated ASR in pipeline

data_size=$1

# Finetune ASR on the pseudolabeled data
subset=train_${data_size}_plabeled_asr
. scripts/asr/finetune.sh manifest/asr/ ${subset}

# Evaluate pipeline model on NER
. scripts/pipeline/eval.sh w2v2_base_${subset} deberta-base_raw_finetune dev combined t3/3