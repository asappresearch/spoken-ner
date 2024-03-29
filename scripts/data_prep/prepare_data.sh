voxpopuli_data_dir=$1
slue_voxpopuli_dir=$2
manifest_dir=$3

# prepare slue_voxpopuli manifest files
python -m slue_toolkit.prepare.prepare_voxpopuli create_manifest \
--data_dir $slue_voxpopuli_dir \
--manifest_dir $manifest_dir 

mkdir -p $manifest_dir/asr
for split in fine-tune dev; do
	for ext in tsv wrd ltr; do
		cp $manifest_dir/${split}.${ext} $manifest_dir/asr/
	done
done

# prepare dictionary files for ASR
for label in ltr wrd; do
    python -m slue_toolkit.prepare.create_dict \
    $manifest_dir/asr/fine-tune.${label} \
    $manifest_dir/asr/dict.${label}.txt
done

# prepare manifest files for external unannotated data
python spoken_ner/prepare/prepare_voxpopuli_ext_data.py prep_unlabeled_data_files \
--voxpopuli_data_dir $voxpopuli_data_dir \
--slue_voxpopuli_dir $slue_voxpopuli_dir \
--manifest_dir $manifest_dir

for target_size in 100 500; do
	python spoken_ner/prepare/prepare_voxpopuli_ext_data.py generate_data_subset \
	--manifest_dir $manifest_dir \
	--target_size $target_size
done
