The checkpoints for chosen model checkpoints can be found at the respective links below. 

#### Dictionary files
1. [ASR](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/asr/dict_files/dict.ltr.txt)
2. [E2E NER](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/e2e_ner/dict_files/dict.raw.ltr.txt)

### Baseline models

Method | checkpoint
----------- | ----------- 
ASR | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/asr/w2v2_base_fine-tune.pt)
E2E NER | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/e2e_ner/w2v2_base_fine-tune.pt)
text NER | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/text_ner/deberta-base_raw_fine-tune.pt)

### Improved pipeline
These are trained on 100 hours of external  data.
External data type      | Method | checkpoint
----------- | ----------- | ----------- 
Un-Sp | SelfTrain-ASR | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/asr/w2v2_base-train_100h_plabeled_asr.pt)
Un-Txt | SelfTrain-txtNer | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/text_ner/deberta-base_raw_train_100h_plabeled_text_ner.pt)
Sp-Txt | Pre-ASR | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/asr/w2v2_base_train_100h.pt)

### Improved E2E NER
These are trained on 100 hours of external data.
External data type      | Method | checkpoint
----------- | ----------- | ----------- 
Un-Sp | Distill-Pipeline | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/e2e_ner/w2v2_base_train_100h_plabeled_ppl.pt)
Sp-Txt | Distill-txtNER | [Link](https://public-dataset-model-store.awsdev.asapp.com/fwu/spoken-ner/public/e2e_ner/w2v2_base_train_100h_plabeled_e2e_ner.pt)
