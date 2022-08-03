# Improve pipeline: external unlabeled text
# Improve text NER module by training on pseudolabeled text NER data tagged using baseline text NER

data_size=$1

# Finetune text NER on the pseudolabeled data
subset=train_${data_size}_plabeled_text_ner
. scripts/text_ner/finetune.sh manifest/ $subset

# Evaluate pipeline model on NER
. scripts/pipeline/eval.sh w2v2_base_fine-tune deberta_base_${subset}_raw dev combined t3/3