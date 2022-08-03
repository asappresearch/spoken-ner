# Improve E2E: external unlabeled speech
# Improve E2E NER module by training on pseudolabeled spoken NER data tagged using pipeline model

data_size=$1

# Train E2E NER on pseudolabeled data
subset_name=train_${data_size}_plabeled_ppl
. scripts/asr/finetune.sh manifest/e2e_ner/ $subset_name

# Evaluate
. scripts/e2e_ner/eval.sh w2v2_base_$subset_name dev combined manifest vp_ner_${subset_name}/4