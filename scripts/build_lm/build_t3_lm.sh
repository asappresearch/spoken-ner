kenlm_dir=$1
data_dir=$2

if ! [ -d ${data_dir}/TEDLIUM_release-3 ]; then
    mkdir -p ${data_dir}
    cd ${data_dir}
    echo "download tedlium 3"
    wget https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz -O TEDLIUM_release-3.tgz 
    
    echo "extract tedlium 3"
    tar -zxvf TEDLIUM_release-3.tgz TEDLIUM_release-3/LM
    cd ..
    rm ${data_dir}/TEDLIUM_release-3.tgz
fi


if ! [ -f ${data_dir}/TEDLIUM_release-3/LM/all_text.en ]; then
    echo "combine lm corpus"
    cat ${data_dir}/TEDLIUM_release-3/LM/*.gz > ${data_dir}/TEDLIUM_release-3/LM/all_text.en.gz
    gunzip ${data_dir}/TEDLIUM_release-3/LM/all_text.en.gz
fi
echo "create ngram LM"
bash scripts/create_ngram.sh ${kenlm_dir} ${data_dir}/TEDLIUM_release-3/LM/all_text.en save/kenlm/t3 3

