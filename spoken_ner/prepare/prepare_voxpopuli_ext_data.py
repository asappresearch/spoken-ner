"""
Prepare external data processed files (for Fairseq and HugginFace) 
"""

import fire
import numpy as np
import os
import pandas as pd
import random
import shutil
import soundfile as sf
import pandas as pd
from tqdm import tqdm

from slue_toolkit import generic_utils as utils
from slue_toolkit.prepare import data_utils


def generate_data_subset(manifest_dir, target_size=100):
    """
    Prepare unlabeled data

    target_size: number of hours
    """
    n_secs = 0
    fs = 16000
    in_data = {}
    out_data = {"tsv": [], "wrd": [], "ltr": []}
    text_ner_tsv_str = ""
    save_dir = os.path.join(manifest_dir, "e2e_ner")

    for key in out_data:
        in_data[key] = utils.read_lst(
            os.path.join(manifest_dir, "e2e_ner", f"train_all_unlabeled.{key}")
        )
        if key == "tsv":
            out_data[key].append(in_data[key][0])

    permuted_idx_lst = np.arange(len(in_data["wrd"]))
    random.shuffle(permuted_idx_lst)

    for item_idx in permuted_idx_lst:
        # e2e_ner and asr manifest files
        for key in out_data:
            if key == "tsv":
                tsv_line = in_data[key][item_idx + 1]
                out_data[key].append(tsv_line)
                n_secs += int(tsv_line.split("\t")[-1]) / fs
            else:
                out_data[key].append(in_data[key][item_idx])
        # text_ner manifest files
        text_ner_tsv_str += (
            "\tO\n".join(in_data["wrd"][item_idx].split(" ")) + "\tO\n\n"
        )

        if (n_secs / 3600) > target_size:
            break

    for key, data in out_data.items():
        utils.write_to_file(
            "\n".join(data),
            os.path.join(save_dir, f"train_{target_size}h_unlabeled.{key}"),
        )
        utils.write_to_file(
            "\n".join(data),
            os.path.join(
                save_dir.replace("e2e_ner", "asr"),
                f"train_{target_size}h_unlabeled.{key}",
            ),
        )
    utils.write_to_file(
        text_ner_tsv_str,
        os.path.join(
            save_dir.replace("e2e_ner", "text_ner"),
            f"train_{target_size}h_unlabeled.tsv",
        ),
    )
    print(f"Manifest files for {np.round(n_secs/3600, 1)} hour subset created")


def convert_plabeled_to_nlp_format(data_sfx="all"):
    """
    For training the transformer model on plabeled data || NLP self-training
    """

    def update_sent_lst(out_wrd_pair_lst, phrase_lst, entity_tag):
        for idx, word in enumerate(phrase_lst):
            if idx == 0 and entity_tag != "O":
                out_wrd_pair_lst.append("\t".join([word, "B-" + entity_tag]))
            elif entity_tag != "O":
                out_wrd_pair_lst.append("\t".join([word, "I-" + entity_tag]))
            else:
                out_wrd_pair_lst.append("\t".join([word, entity_tag]))

    tagged_sent_lst = read_lst(
        os.path.join(SPEECH_NER_DATA_DIR, f"train_{data_sfx}_plabeled_nlp.wrd")
    )
    out_str = ""
    num_illegal_assigments = 0
    for line in tagged_sent_lst:
        is_entity, phrase_lst, out_wrd_pair_lst = False, [], []
        wrd_lst = line.split(" ")
        for wrd in wrd_lst:
            if wrd in DATA_OBJ.spl_char_lst:
                if (
                    is_entity
                ):  # a new entity begun before completion of the previous entity
                    phrase_lst = []  # discard the ongoing entity
                    num_illegal_assigments += 1
                is_entity = True
                entity_tag = utils.spl_char_to_entity[wrd]
            elif wrd == DATA_OBJ.end_char:
                if is_entity:
                    if len(phrase_lst) > 0:
                        update_sent_lst(out_wrd_pair_lst, phrase_lst, entity_tag)
                    else:
                        num_illegal_assigments += 1
                    phrase_lst = []
                    is_entity = False
                else:
                    num_illegal_assigments += 1
            else:
                if is_entity:
                    phrase_lst.append(wrd)
                else:
                    update_sent_lst(out_wrd_pair_lst, [wrd], "O")
        out_str += "\n".join(out_wrd_pair_lst) + "\n\n"
    print(
        "%d illegal assignments | %d sentences"
        % (num_illegal_assigments, len(tagged_sent_lst))
    )
    write_to_file(
        out_str, os.path.join(TEXT_NER_DATA_DIR, f"train_{data_sfx}_plabeled")
    )


def prep_unlabeled_data_files(voxpopuli_data_dir, slue_voxpopuli_dir, manifest_dir):
    """
    Process all unlabeled data into fairseq and huggingface formats

    voxpopuli_data_dir: path to voxpopuli data (downloaded using the voxpopuli repo)
    slue_voxpopuli_dir: path to slue-voxpopuli data (downloaded using the slue-toolkit repo)
    manifest_dir: directory where the processed files will be saved
    """

    # create output directories
    e2e_ner_save_dir = os.path.join(manifest_dir, "e2e_ner")
    os.makedirs(e2e_ner_save_dir, exist_ok=True)
    asr_save_dir = os.path.join(manifest_dir, "asr")
    os.makedirs(asr_save_dir, exist_ok=True)
    text_ner_save_dir = os.path.join(manifest_dir, "text_ner")
    os.makedirs(text_ner_save_dir, exist_ok=True)

    text_ner_tsv_fn = os.path.join(text_ner_save_dir, "train_all_unlabeled.tsv")
    e2e_tsv_fn = os.path.join(e2e_ner_save_dir, "train_all_unlabeled.tsv")
    wrd_fn = os.path.join(e2e_ner_save_dir, "train_all_unlabeled.wrd")
    ltr_fn = os.path.join(e2e_ner_save_dir, "train_all_unlabeled.ltr")

    # read raw formats
    audio_data_dir = os.path.join(voxpopuli_data_dir, "transcribed_data/en")
    all_train_data_df = pd.read_csv(
        os.path.join(audio_data_dir, "asr_train.tsv"), sep="\t"
    )
    tot_num_samples = len(all_train_data_df["id"].array)
    slue_finetune_data_df = pd.read_csv(
        os.path.join(slue_voxpopuli_dir, "slue-voxpopuli_fine-tune.tsv"), sep="\t"
    )

    # write processed data
    with open(wrd_fn, "w") as f_wrd, open(ltr_fn, "w") as f_ltr, open(
        e2e_tsv_fn, "w"
    ) as f_e2e_tsv, open(text_ner_tsv_fn, "w") as f_txt_ner_tsv:
        print(audio_data_dir, file=f_e2e_tsv)
        for data_sample in tqdm(all_train_data_df.iterrows()):
            utt_id = data_sample[1].id
            normalized_text = data_sample[1].normalized_text
            if (
                utt_id not in slue_finetune_data_df["id"].array
                and normalized_text != ""
                and not pd.isna(normalized_text)
            ):
                entity_pair_str = data_utils.prep_text_ner_tsv(
                    normalized_text,
                    "None",  # no labels
                    "combined",  # value of this arg doesn't matter as this is unlabeled data
                )
                filtered_entity_pair_str = ""
                # filtering <empty string, "O"> tag pairs (corner case)
                for wrd in entity_pair_str.split("\n"):
                    if (
                        wrd != ""
                    ):  # handles last two empty strings created because of "\n\n"
                        try:
                            assert wrd[-2:] == "\tO"
                        except:
                            import pdb

                            pdb.set_trace()
                        if wrd[0] != "\t":
                            filtered_entity_pair_str += wrd + "\n"
                assert len(filtered_entity_pair_str) > 0
                filtered_entity_pair_str += "\n"

                print(filtered_entity_pair_str, file=f_txt_ner_tsv, end="")

                wrd_str, ltr_str = data_utils.prep_e2e_ner_files(
                    filtered_entity_pair_str,
                    "combined",  # value of this arg doesn't matter as this is unlabeled data
                )
                print(wrd_str, file=f_wrd)
                print(ltr_str, file=f_ltr)

                audio_pth = os.path.join(utt_id[:4], utt_id + ".ogg")
                audio, _ = sf.read(os.path.join(audio_data_dir, audio_pth))
                print("\t".join([audio_pth, str(len(audio))]), file=f_e2e_tsv)

    shutil.copyfile(e2e_tsv_fn, e2e_tsv_fn.replace("e2e_ner", "asr"))
    shutil.copyfile(wrd_fn, wrd_fn.replace("e2e_ner", "asr"))
    shutil.copyfile(ltr_fn, ltr_fn.replace("e2e_ner", "asr"))

    print(f"Manifest files created for all unlabeled data, saved at {manifest_dir}")


if __name__ == "__main__":
    fire.Fire()
