data_size=$1
kenlm_build_bin=$2

manifest_dir=manifest
ckpt_dir=save

# NER pseudolabels using E2E model
# 1. Generate pseudolabels
. scripts/e2e_ner/decode.sh w2v2_base_fine-tune train_${data_size}_unlabeled manifest vp_ner_fine-tune
# 2. Reformat pseudolabels to be compatible for further training
python spoken_ner/prepare/reformat_decoded_output.py fairseq \
--manifest_dir ${manifest_dir}/ \
--model_dir ${ckpt_dir}/ \
--train_model w2v2_base_fine-tune \
--plabel_data train_${data_size} \
--lm vp_ner_fine-tune \
--task e2e_ner
# 3. Train language model on the pseudolabels
subset_name="train_${data_size}_plabeled_e2e_ner"
. scripts/build_lm/create_ngram.sh $kenlm_build_bin ${manifest_dir}/e2e_ner/${subset_name}.raw.wrd ${ckpt_dir}/kenlm/vp_ner_${subset_name} 4


# ASR pseudolabels using ASR model
# 1. Generate pseudolabels
python -m slue_toolkit.eval.eval_w2v eval_ctc_model ${ckpt_dir}/asr/w2v2_base_fine-tune --data ${manifest_dir}/ --subset train_${data_size}_unlabeled --lm t3/3
# 2. Reformat pseudolabels to be compatible for further training
python spoken_ner/prepare/reformat_decoded_output.py fairseq \
--manifest_dir ${manifest_dir}/ \
--model_dir ${ckpt_dir}/ \
--train_model w2v2_base_fine-tune \
--plabel_data train_${data_size} \
--lm t3/3 \
--task asr


# NER pseudolabels using pipeline model
# 1. Generate pseudolabels
. scripts/pipeline/tag.sh w2v2_base_fine-tune ${ckpt_dir}/text_ner/deberta-base_raw_fine-tune train_${data_size}_unlabeled raw t3/3
# 2. Reformat pseudolabels to be compatible for further training
python spoken_ner/prepare/reformat_decoded_output.py text_ner \
--manifest_dir ${manifest_dir}/ \
--plabel_data train_${data_size} \
--task text_ner
# 3. Train language model on the pseudolabels
subset_name="train_${data_size}_plabeled_ppl"
. scripts/build_lm/create_ngram.sh $kenlm_build_bin ${manifest_dir}/e2e_ner/${subset_name}.raw.wrd ${ckpt_dir}/kenlm/vp_ner_${subset_name} 4


# NER pseudolabels using text NER model
# 1. Generate pseudolabels
. scripts/text_ner/tag.sh ${ckpt_dir}/text_ner/deberta-base_raw_fine-tune train_${data_size} 
# 2. Reformat pseudolabels to be compatible for further training
python spoken_ner/prepare/reformat_decoded_output.py text_ner \
--manifest_dir ${manifest_dir}/ \
--plabel_data train_${data_size} \
--task text_ner
# 3. Train language model on the pseudolabels
subset_name="train_${data_size}_plabeled_text_ner"
. scripts/build_lm/create_ngram.sh $kenlm_build_bin ${manifest_dir}/e2e_ner/${subset_name}.raw.wrd ${ckpt_dir}/kenlm/vp_ner_${subset_name} 4

