# Improve pipeline: external speech-text data
# Train ASR on external transcribed speech and then use the updated ASR in pipeline 
data_size=$1

# Finetune ASR on larger data
subset=train_${data_size}_unlabeled
. scripts/asr/finetune.sh manifest/asr/ ${subset}

# Evaluate pipeline model on NER
. scripts/pipeline/eval.sh w2v2_base_${subset} deberta_base dev combined t3/3
