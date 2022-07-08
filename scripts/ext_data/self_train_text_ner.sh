# Improve pipeline: external unlabeled text
# Improve text NER module by training on pseudolabeled text NER data tagged using baseline text NER

data_size=$1

. scripts/text_ner/ft_text_ner.sh manifest/text_ner train_${data_size}_plabeled_text_ner

. scripts/pipeline/eval.sh fine-tune deberta_base_train_${data_size}_plabeled_text_ner dev combined t3