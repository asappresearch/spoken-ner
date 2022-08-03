"""
Inference on unlabaled data
"""

import fire
import os
import numpy as np

from torch.utils.data import DataLoader

import slue_toolkit.text_ner.ner_deberta_modules as NDM
from slue_toolkit.generic_utils import write_to_file

import sys

sys.path.insert(0, "spoken_ner/prepare")
from reformat_decoded_output import reformat_transformers_to_fairseq


def reformat(wrd_lst, label_lst):
    wrd_tag_lst = []
    for idx, wrd in enumerate(wrd_lst):
        wrd_tag_lst.append(wrd + "\t" + label_lst[idx])
    return wrd_tag_lst


def inference(
    data_dir,
    model_dir,
    model_type,
    eval_asr=False,
    eval_subset="dev",
    train_label="raw",
    eval_label="combined",
    lm="t3/3",
    asr_model_type="w2v2-base",
):
    data_obj = NDM.DataSetup(data_dir, model_type)
    if eval_asr:
        lm = lm.replace("/", "_")
        split_name = f"{eval_subset}-{asr_model_type}-asr-{lm}.tsv"
        write_fn_pfx = "ppl"
    else:
        split_name = eval_subset + "_unlabeled"
        write_fn_pfx = "text_ner"
    write_fn = os.path.join(
        data_dir, "..", "e2e_ner", f"{write_fn_pfx}_plabeled_{eval_subset}"
    )
    val_texts, _, _, _, val_dataset = data_obj.prep_data(split_name, eval_label, False)

    eval_obj = NDM.Eval(data_dir, model_dir, train_label, eval_label, eval_asr)

    all_reformated_sent = []
    data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(eval_obj.device)
        attention_mask = batch["attention_mask"].to(eval_obj.device)
        outputs = eval_obj.model(input_ids, attention_mask=attention_mask)
        labels = batch["labels"].detach().numpy()
        predictions = np.argmax(outputs.logits.cpu().detach().numpy(), axis=2)
        entity_predictions = [
            [eval_obj.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        wrd_lst = val_texts[idx]
        assert len(entity_predictions[0]) == len(wrd_lst)
        sent_lst = reformat(wrd_lst, entity_predictions[0])
        wrd_sent = reformat_transformers_to_fairseq(sent_lst, data_dir, eval_label)
        all_reformated_sent.append(wrd_sent)
        all_reformated_sent_ltr = [
            " ".join(list(sent.replace(" ", "|"))) + " |"
            for sent in all_reformated_sent
        ]

    write_to_file("\n".join(all_reformated_sent_ltr), write_fn + ".ltr")
    write_to_file("\n".join(all_reformated_sent), write_fn + ".wrd")
    print(f"{str(len(all_reformated_sent))} sentences tagged")


if __name__ == "__main__":
    fire.Fire(inference)
