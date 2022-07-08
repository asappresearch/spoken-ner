# Improve E2E: external unlabeled speech
# Improve E2E NER module by training on pseudolabeled spoken NER data tagged using baseline E2E NER

data_size=$1

subset_name=train_${data_size}_plabeled_e2e_ner
. scripts/e2e_ner/ft_e2e_ner.sh manifest/e2e_ner/ $subset_name

. scripts/e2e_ner/eval_e2e_ner.sh $subset_name dev combined manifest/e2e_ner $subset_name
