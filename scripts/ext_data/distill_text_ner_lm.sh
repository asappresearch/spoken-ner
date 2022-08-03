# Improve e2e: external speech-text data
# Decode E2E NER using improved LM trained on psedolabeled data from text NER 

data_size=$1

subset_name=train_${data_size}_plabeled_text_ner

# Evaluate
. scripts/e2e_ner/eval.sh w2v2_base_fine-tune dev combined manifest vp_ner_${subset_name}/4