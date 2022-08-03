"""
Create manifest files for outputs from different modules
"""

import fire
import os

from slue_toolkit import generic_utils as utils
from slue_toolkit.generic_utils import read_lst, write_to_file


def remove_annotations(line):
    spl_char_lst = list(utils.spl_char_to_entity.keys()) + [utils.end_char]
    for spl_char in spl_char_lst:
        line = line.replace(spl_char, "")
    line = line.strip()
    line = line.replace("   ", " ")
    line = line.replace("  ", " ")
    return line


def reorder_outputs(
    plabel_data,
    manifest_dir,
    lm=None,
    model_dir=None,
    task_name=None,
    text_ner=False,
    fq_to_fq=False,
):
    """
    Reorder the decoded output to match the same order as the tsv files

    task_name: ppl | text_ner
    """

    def get_data_lists_fq(data_ref, data_hyp, orig_data, orig_data_tsv):
        out_data_wrd, out_data_ltr, out_data_tsv = [], [], []
        num_discarded = 0
        for sample_idx, line in enumerate(data_ref):
            try:
                idx_new = orig_data.index(line)
            except:
                num_discarded += 1
                continue
            orig_data[idx_new] = -1  # to ensure that it's not chosen again
            out_data_wrd.append(data_hyp[sample_idx])
            out_data_ltr.append(
                " ".join(list(data_hyp[sample_idx].replace(" ", "|"))) + " |"
            )
            out_data_tsv.append(orig_data_tsv[idx_new + 1])
        return out_data_wrd, out_data_ltr, out_data_tsv, num_discarded

    def get_data_lists_nlp(data_hyp, orig_data, orig_data_tsv):
        out_data_wrd, out_data_ltr, out_data_tsv = [], [], []
        num_discarded = 0
        for _, line in enumerate(data_hyp):
            filtered_line = remove_annotations(line)
            try:
                idx_new = orig_data.index(filtered_line)
            except:
                num_discarded += 1
                continue
            orig_data[idx_new] = -1  # to ensure that it's not chosen again
            out_data_wrd.append(line)
            out_data_ltr.append(" ".join(list(line.replace(" ", "|"))) + " |")
            out_data_tsv.append(orig_data_tsv[idx_new + 1])
        return (
            out_data_wrd,
            out_data_ltr,
            out_data_tsv,
            num_discarded + len(orig_data) - len(data_hyp),
        )

    assert text_ner or fq_to_fq
    assert not (text_ner and fq_to_fq)

    split_name = f"{plabel_data}_unlabeled"
    if text_ner:
        data_hyp = read_lst(
            os.path.join(
                manifest_dir, "e2e_ner", f"{plabel_data}_plabeled_{task_name}.wrd"
            )
        )
        orig_data = read_lst(os.path.join(manifest_dir, "e2e_ner", f"{split_name}.wrd"))
        orig_data_tsv = read_lst(
            os.path.join(manifest_dir, "e2e_ner", f"{split_name}.tsv")
        )
        return get_data_lists_nlp(data_hyp, orig_data, orig_data_tsv)
    elif fq_to_fq:
        decoded_data_dir = os.path.join(model_dir, "decode", lm.replace("/", "_"))
        orig_data = read_lst(os.path.join(manifest_dir, f"{split_name}.wrd"))
        orig_data_tsv = read_lst(os.path.join(manifest_dir, f"{split_name}.tsv"))

        assert len(orig_data_tsv) == len(orig_data) + 1

        data_hyp_all = read_lst(
            os.path.join(
                decoded_data_dir, "hypo.word-checkpoint_best.pt-" + split_name + ".txt"
            )
        )
        data_ref_all = read_lst(
            os.path.join(
                decoded_data_dir, "ref.word-checkpoint_best.pt-" + split_name + ".txt"
            )
        )
        assert len(data_ref_all) == len(data_hyp_all)
        assert len(data_ref_all) == len(orig_data)
        # remove empty outputs
        data_hyp = [
            line.split(" (None-")[0]
            for line in data_hyp_all
            if len(line.split(" (None-")) > 1
        ]
        data_ref = [
            line.split(" (None-")[0]
            for idx, line in enumerate(data_ref_all)
            if len(data_hyp_all[idx].split(" (None-")) > 1
        ]
        return get_data_lists_fq(data_ref, data_hyp, orig_data, orig_data_tsv)


def process_fairseq_output(manifest_dir, model_dir, train_model, plabel_data, lm, task):
    """
    Process the output of e2e asr or e2e ner modules into fairseq compatible
    manifest files.
    """
    write_fn = os.path.join(manifest_dir, task, f"{plabel_data}_plabeled_{task}")
    out_data_tsv = read_lst(os.path.join(manifest_dir, task, "fine-tune.tsv"))
    out_data_wrd = read_lst(os.path.join(manifest_dir, task, "fine-tune.wrd"))
    out_data_ltr = read_lst(os.path.join(manifest_dir, task, "fine-tune.ltr"))
    data_wrd, data_ltr, data_tsv, num_utt_missed = reorder_outputs(
        plabel_data,
        manifest_dir,
        lm=lm,
        model_dir=os.path.join(model_dir, task, train_model),
        fq_to_fq=True,
    )

    out_data_tsv.extend(data_tsv)
    out_data_wrd.extend(data_wrd)
    out_data_ltr.extend(data_ltr)
    write_to_file("\n".join(out_data_wrd), write_fn + ".wrd")
    write_to_file("\n".join(out_data_ltr), write_fn + ".ltr")
    write_to_file("\n".join(out_data_tsv), write_fn + ".tsv")
    print(
        "%d samples written to %s (%d discarded)"
        % (len(out_data_wrd), write_fn, num_utt_missed)
    )


def reformat_fairseq_to_transformers(manifest_dir, plabel_data):
    def update_sent_lst(out_wrd_pair_lst, phrase_lst, entity_tag):
        for idx, word in enumerate(phrase_lst):
            if idx == 0 and entity_tag != "O":
                out_wrd_pair_lst.append("\t".join([word, "B-" + entity_tag]))
            elif entity_tag != "O":
                out_wrd_pair_lst.append("\t".join([word, "I-" + entity_tag]))
            else:
                out_wrd_pair_lst.append("\t".join([word, entity_tag]))

    spl_char_to_entity = utils.spl_char_to_entity
    tagged_sent_lst = read_lst(
        os.path.join(manifest_dir, "e2e_ner", f"{plabel_data}_plabeled_text_ner.wrd")
    )
    spl_char_lst = list(spl_char_to_entity.keys())
    out_str = ""
    num_illegal_assigments = 0
    for line in tagged_sent_lst:
        is_entity, phrase_lst, out_wrd_pair_lst = False, [], []
        wrd_lst = line.split(" ")
        for wrd in wrd_lst:
            if wrd in spl_char_lst:
                if (
                    is_entity
                ):  # a new entity begun before completion of the previous entity
                    phrase_lst = []  # discard the ongoing entity
                    num_illegal_assigments += 1
                is_entity = True
                entity_tag = spl_char_to_entity[wrd]
            elif wrd == utils.end_char:
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
        out_str,
        os.path.join(manifest_dir, "text_ner", f"{plabel_data}_plabeled_text_ner.tsv"),
    )


def process_text_ner_output(manifest_dir, plabel_data, task):
    """
    Process the output of text_ner module into both fairseq and
    transformers compatible manifest files.
    """
    # this file will be rewritten
    write_fn = os.path.join(
        manifest_dir, "e2e_ner", f"{plabel_data}_plabeled_{task}.wrd"
    )
    out_data_tsv = read_lst(os.path.join(manifest_dir, "e2e_ner", "fine-tune.tsv"))
    out_data_wrd = read_lst(os.path.join(manifest_dir, "e2e_ner", "fine-tune.wrd"))
    out_data_ltr = read_lst(os.path.join(manifest_dir, "e2e_ner", "fine-tune.ltr"))
    data_wrd, data_ltr, data_tsv, num_utt_missed = reorder_outputs(
        plabel_data, f"{manifest_dir}/e2e_ner", task_name=task, fq_to_fq=True
    )
    out_data_tsv.extend(data_tsv)
    out_data_wrd.extend(data_wrd)
    out_data_ltr.extend(data_ltr)
    write_to_file("\n".join(out_data_wrd), write_fn + ".wrd")
    write_to_file("\n".join(out_data_ltr), write_fn + ".ltr")
    write_to_file("\n".join(out_data_tsv), write_fn + ".tsv")
    print(
        "%d samples written to %s (%d discarded)"
        % (len(out_data_wrd), write_fn, num_utt_missed)
    )

    if task == "text_ner":
        reformat_fairseq_to_transformers(manifest_dir, plabel_data)


def reformat_transformers_to_fairseq(sent_str, label_type):
    """
    Process sentence from the transformers .tsv format to fairseq .wrd file
    """
    cnt = 0
    entity, do_else = False, False
    out_sent = []
    label_map_dct = getattr(utils, f"{label_type}_entity_to_spl_char")
    for line in sent_str:
        wrd, tag = line.split("\t")
        if tag != "O":
            cnt += 1
            if entity:
                if tag[:2] == "I-":
                    # assert tag == "I-"+prev_tag
                    if tag == "I-" + prev_tag:
                        curr_wrd += " " + wrd
                        do_else = False
                    else:
                        do_else = True
                if tag[:2] == "B-" or do_else:
                    out_sent.append(curr_wrd + " " + utils.end_char)
                    curr_wrd = label_map_dct[tag[2:]] + " " + wrd
                    do_else = False
                prev_tag = tag[2:]
            else:
                # assert tag[:2] == "B-"
                prev_tag = tag[2:]
                curr_wrd = label_map_dct[tag[2:]] + " " + wrd
                entity = True
        else:
            if entity:
                curr_wrd += " " + utils.end_char + " " + wrd
                entity = False
            else:
                curr_wrd = wrd
        if not entity:
            out_sent.append(curr_wrd)
    if entity:
        out_sent.append(curr_wrd + " " + utils.end_char)

    return " ".join(out_sent)


if __name__ == "__main__":
    fire.Fire(
        {
            "fairseq": process_fairseq_output,
            "text_ner": process_text_ner_output,
        }
    )
