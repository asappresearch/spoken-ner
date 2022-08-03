# spoken-ner
This codebase helps replicate the results from our paper [On the Use of External Data for Spoken Named Entity Recognition](https://arxiv.org/pdf/2112.07648.pdf) published at NAACL 2022.

Fine-tuned models reported in the paper can be found [here](https://github.asapp.dev/ASAPPinc/spoken-ner/blob/main/checkpoints/README.md).

# Usage

## 1. Dependencies 
This repo uses libraries from the [slue-toolkit](https://github.com/asappresearch/slue-toolkit) and [voxpopuli](https://github.com/facebookresearch/voxpopuli#getting-data) repositories.

Please make sure you have both these repositories cloned with all the necessary dependencies installed in your working environment.

Note: Last checked with fairseq [commit 5307a0e](https://github.com/facebookresearch/fairseq/tree/5307a0e078d7460003a86f4e2246d459d4706a1d).

## 2. Data preparation
### 2a. Download data
This script will download voxpopuli (~80Gb of raw audio and transcripts) and slue-voxpopuli (NER annotations for 25 hours voxpopuli) data at `$vp_data_dir/voxpopuli` and `$vp_data_dir/slue-voxpopuli` respectively.
```
. scripts/data_prep/download_datasets.sh $vp_data_dir
```

### 2b. Prepare manifest files
Prepare manifest files for labeled and external unlabeled data (100h and 500h subsets)
```
. scripts/data_prep/prepare_data.sh $vp_data_dir/voxpopuli $vp_data_dir/slue-voxpopuli manifest/
```

## 3. Establish baselines
These models will be used further for pseudo-labeling in the next step. Alternatively, you can download these finetuned models and language models directly from the link mentioned above.

### 3a. Baseline models
Follow [slue-toolkit](https://github.com/asappresearch/slue-toolkit) to [finetune the NER models using slue-voxpopuli](https://github.com/asappresearch/slue-toolkit/blob/main/baselines/ner/README.md) (no external data). 

Note: slue-toolkit expects the manifest files at `slue-toolkit/manifest`, so make sure to appropriately symlink the files, 

Make sure that all the new model directories are created/moved/linked such that those can be found under the `spoken-ner/save/` directory. The following naming convention is expected:

    (i) `spoken-ner/save/asr/w2v2_base_fine-tune` -> `slue-toolkit/save/e2e_ner/w2v2-base-vp` 

    (ii) `spoken-ner/save/e2e_ner/w2v2_base_fine-tune` -> `slue-toolkit/save/e2e_ner/w2v2-base` 

    (iii) `spoken-ner/save/text_ner/deberta-base_raw_fine-tune` -> `slue-toolkit/save/text_ner/deberta-base_raw` 

### 3b. Language models
Train a text LM and a text NER LM using the commands below. The trained models will be saved under `save/kenlm/` directory. Set the `kenlm_build_bin` to the path of your kenlm build folder (e.g., /home/user/kenlm/build/bin),

1. 3-gram LM trained on on TEDLIUM-3 corpus (this will temporarily download 50 Gb TEDLIUM-3 corpus, but retaining only the text part for LM training at `save_dir`):  
```
. scripts/build_lm/build_t3_lm.sh $kenlm_build_bin $save_dir 
```

2. 4-gram LM trained on NER-annotated voxpopuli text data 
```
. scripts/build_lm/create_ngram.sh $kenlm_build_bin manifest/e2e_ner/fine-tune.raw.wrd save/kenlm/vp_ner_fine-tune 4
```

## 4. Decode external data
Decode the data prepared in the previous step using the baseline models to create pseudo-labeled data and train language models on the psuedolabeled NER data.
```
. scripts/ext_data/decode_ext_data.sh $data_size $kenlm_build_bin
```

Each section in the script decodes the unlabeled data, reformats it for further finetuning, and trains a language model on the decoded text (in case of NER). There are four such sections: (i) E2E NER, (ii) ASR, (iii) pipeline NER, (iv) text NER.

Execution tip #1: Each decoding section is independent of each other and should be run separately.

Execution tip #2: Each decoding section can be split further to decode subsets of the whole `data_size` data parallely. This will require splitting the manifest files before processing and appending the decoded outputs before running the reformatting code.

## 5. Fine-tune models using decoded data
Refer to the paper for details on the nomenclature used in the tables and for more details regarding each method.

### 5a. Improve pipeline models
External data type      | Method | Target model | command
----------- | ----------- | ----------- | -----------
Un-Sp | SelfTrain-ASR | ASR | `. scripts/ext_data/self_train_asr.sh $data_size`
Un-Txt | SelfTrain-txtNer | text NER | `. scripts/ext_data/self_train_text_ner.sh $data_size`
Sp-Txt | Pre-ASR | ASR | `. scripts/ext_data/pre_asr.sh $data_size`

### 5b. Improve E2E models
 External data type      | Method | Labeling model | Target model | LM for decoding | command
 ----------- | ----------- | ----------- | ----------- | ----------- | ----------
 Un-Sp      | SelfTrain-E2E       | E2E-NER | E3E-NER | plabel 4-gram | `. scripts/ext_data/self_train_e2e_ner.sh $data_size`
 Un-Sp   | Distill-Pipeline      | E2E-NER | E2E-NER | plabel 4-gram | `. scripts/ext_data/distill_pipeline.sh $data_size`
 Un-Txt | Distill-txtNER-lm | text NER | n/a | plabel 4-gram   | `. scripts/ext_data/distill_text_ner_lm.sh $data_size`
 Sp-Txt | Distill-txtNER | text NER | E2E-NER | plabel 4-gram |  `. scripts/ext_data/distill_text_ner.sh $data_size`
 Sp-Txt | Pre-ASR | n/a | n/a | ftune 4-gram | `. scripts/ext_data/pre_asr_e2e_ner.sh $data_size`

## 6. Error analysis
The following script categorizes the errors into different categories and saves examples from individual category as well as gives error distribution.

This will require installation of the following additional libraries: `seaborn`, `nltk`.

Following are the examples:
- `. scripts/analysis/e2e_ner.sh w2v2_base_train_100h_plabeled_text_ner vp_ner_train_100h_plabeled_text_ner/4`
- `. scripts/analysis/pipeline.sh deberta-base_fine-tune_raw w2v2_base_train_100h_plabeled_asr`
- `. scripts/analysis/text_ner.sh deberta-base_fine-tune_raw`

## 7. Additional experimentation possible with this codebase
Experiments with different labeling and target models are possible with this codebase, for example if you wish to do multiple iterations of self-training of use a tagger different from the baseline models. Make sure to pass the appropriate model names in the scripts noted in [section 5](https://github.asapp.dev/ASAPPinc/spoken-ner#5-fine-tune-models-using-decoded-data).
