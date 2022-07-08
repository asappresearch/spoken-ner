export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

ngpu=1
seed=1

manifest_dir=$1
train_subset=$2 # fine-tune
pretrained_ckpt=$3

if [ -z "$pretrained_ckpt" ]
then
    save="save/asr/w2v2_base_${train_subset}"
    pretrained_ckpt="save/pretrained/wav2vec_small.pt"
    if ! [ -f $pretrained_ckpt ]; then
        mkdir -p save/pretrained
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -O $pretrained_ckpt
    fi
else
    # pretrained ckpt: ASR trained on external data
    pretrained_ckpt=$pretrained_ckpt
    save="save/asr/w2v2_pre_asr_${train_subset}"
fi

save="save/e2e_ner/w2v2_base_${train_subset}"
pretrained_ckpt="save/pretrained/wav2vec_small.pt"
if ! [ -f $pretrained_ckpt ]; then
    mkdir -p save/pretrained
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -O $pretrained_ckpt
fi

mkdir -p $save
tb_save=${save}/tb_logs


data=`realpath $manifest_dir`
pretrained_ckpt=`realpath $pretrained_ckpt`

config_dir=configs
if [[ "$train_subset" == *"500h"* ]]; then
    config=w2v2_base_ner_500h.yaml
elif [[ "$train_subset" == *"100h"* ]]; then
    config=w2v2_base_ner_100h.yaml
elif [[ "$train_subset" == "fine-tune" ]]; then
    config=w2v2_base_ner_15h.yaml
fi
valid_subset=dev

normalize=false
lr=5e-5
max_tokens=3200000
max_update=20000

fairseq-hydra-train \
    hydra.run.dir=$save \
    hydra.output_subdir=$save \
    common.tensorboard_logdir=$tb_save \
    task.data=$data \
    task.labels="raw.ltr" \
    dataset.train_subset=$train_subset \
    dataset.valid_subset=$valid_subset \
    distributed_training.distributed_world_size=$ngpu \
    common.seed=$seed \
    model.w2v_path="$pretrained_ckpt" \
    optimization.max_update=$max_update \
    dataset.max_tokens=$max_tokens \
    task.normalize=$normalize \
    --config-dir $config_dir \
    --config-name $config