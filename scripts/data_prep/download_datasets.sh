# Ref for vox-populi: https://github.com/facebookresearch/voxpopuli#getting-data
# Ref for slue-voxpopuli: https://github.com/asappresearch/slue-toolkit/blob/main/README.md#datasets

data_dir=$1
voxpopuli_repo_dir=$2

# Download slue-voxpopuli data
if [ -d ${data_dir}/slue-voxpopuli ]; then
    echo "${data_dir}/slue-voxpopuli exists. Skip download & extract."
else
    #1. Download
    tar_file="${data_dir}/slue-voxpopuli_blind.tar.gz"
    if [ -f $tar_file ]; then
        echo "$tar_file exists. Skip download."
    else
        tar_file_url="https://papers-slue.awsdev.asapp.com/slue-voxpopuli_blind.tar.gz"
        wget $tar_file_url -P $data_dir/
    fi

    #2. Extract
    tar -xzvf $tar_file -C $data_dir/
fi

# Download voxpopuli data
if [ -d ${data_dir}/voxpopuli ]; then
    echo "${data_dir}/voxpopuli exists. Skip download & extract."
else
	curr_dir=`pwd`
	cd $voxpopuli_repo_dir
	python -m voxpopuli.download_audios --root ${data_dir}/voxpopuli/ --subset asr
	python -m voxpopuli.get_asr_data --root ${data_dir}/voxpopuli/ --lang en
	cd $curr_dir
fi