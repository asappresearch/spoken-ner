import ast
import editdistance
import fire
import numpy as np
import os

import matplotlib

matplotlib.rcParams.update({"figure.autolayout": True})
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

import nltk
from nltk.corpus import stopwords

FUNC_WRDS = stopwords.words("english")

from slue_toolkit import generic_utils as utils
from slue_toolkit.generic_utils import save_dct, read_lst, load_dct, write_to_file


class ClassifyErrors:
    def __init__(
        self,
        model_dir,
        e2e_ner_lm=None,
        eval_label="combined",
        eval_split="dev",
        asr_model_name=None,
    ):
        self.eval_label = eval_label
        self.eval_split = eval_split
        log_dir = os.path.join(model_dir, "metrics", "error_analysis")
        if "e2e_ner" in log_dir:
            self.model_type = "e2e_ner"
            log_fn = "-".join([eval_split, e2e_ner_lm, eval_label, "standard"])
            self.pred_fn = os.path.join(
                model_dir,
                "decode",
                e2e_ner_lm,
                f"hypo.word-checkpoint_best.pt-{eval_split}.txt",
            )
        elif "text_ner" in log_dir and asr_model_name is None:
            self.model_type = "text_ner"
            log_fn = "-".join([eval_split, "gt-text", eval_label, "standard"])
        else:
            self.model_type = "pipeline"
            log_fn = "-".join(
                [eval_split, "pipeline", asr_model_name, "t3_3", eval_label, "standard"]
            )
            self.pred_fn = os.path.join(
                model_dir,
                "decode",
                "t3_3",
                f"hypo.word-checkpoint_best.pt-{eval_split}.txt",
            )
        self.log_fn = os.path.join(log_dir, log_fn + ".json")
        self.model_dir = model_dir
        self.error_combination = {
            "correct": ["correct"],
            "incorrect asr, incorrect ner": [
                "missed_detection_misspelled",
                "partial_detect_incorrect_tag",
                "misspelled_incorrect_tag",
            ],  # , "over_detection_misspelled"],
            "incorrect asr, correct ner": [
                "partial_detect_correct_tag",
                "misspelled_correct_tag",
            ],
            "correct asr, incorrect ner": [
                "mislabel",
                "partial_detect_correct_asr",
                "missed_detection_correctly_spelled",
            ],
        }  # , \
        # "over_detection_correctly_spelled"]}
        common_error_categories = [
            "correct",
            "mislabel",
            "partial_detect_correct_tag",
            "partial_detect_incorrect_tag",
            "partial_detect_correct_asr",
            "misspelled_correct_tag",
            "misspelled_incorrect_tag",
        ]
        self.gt_category_keys = common_error_categories + [
            "missed_detection_correctly_spelled",
            "missed_detection_misspelled",
        ]
        self.pred_category_keys = common_error_categories + [
            "over_detection_misspelled",
            "over_detection_correctly_spelled",
        ]
        self.all_category_keys = list(
            set(self.gt_category_keys + self.pred_category_keys)
        )

        assert (
            len(
                set(
                    [
                        item
                        for elem in list(self.error_combination.values())
                        for item in elem
                    ]
                )
                - set(self.gt_category_keys)
            )
            == 0
        )
        assert (
            len(
                set(self.gt_category_keys)
                - set(
                    [
                        item
                        for elem in list(self.error_combination.values())
                        for item in elem
                    ]
                )
            )
            == 0
        )

    def remove_annotations(self, line):
        spl_char_lst = list(utils.spl_char_to_entity.keys()) + [utils.end_char]
        for spl_char in spl_char_lst:
            line = line.replace(spl_char, "")
        line = line.strip()
        line = line.replace("   ", " ")
        line = line.replace("  ", " ")
        return line

    def reorder(self, example_lst):
        """
        Reorder predicted text list
        """
        pred_txt_lst = read_lst(self.pred_fn)
        gt_txt_lst = read_lst(
            os.path.join(
                os.path.dirname(self.pred_fn),
                f"ref.word-checkpoint_best.pt-{self.eval_split}.txt",
            )
        )

        pred_txt_lst = [
            self.remove_annotations(line.split(" (None-")[0]) for line in pred_txt_lst
        ]
        gt_txt_lst = [
            self.remove_annotations(line.split(" (None-")[0]) for line in gt_txt_lst
        ]
        example_lst_txt = [line.split("\t")[0] for line in example_lst]
        assert len(example_lst_txt) == len(pred_txt_lst)
        decoded_sent_lst_reordered = [None] * len(pred_txt_lst)
        for idx, line in enumerate(gt_txt_lst):
            assert line != -1
            idx_new = example_lst_txt.index(line)
            example_lst_txt[idx_new] = -1  # to ensure that it's not chosen again
            decoded_sent_lst_reordered[idx_new] = pred_txt_lst[idx]
        return decoded_sent_lst_reordered

    def read_files(self):
        example_lst = load_dct(self.log_fn)[
            "all"
        ]  # list of tab separated gt_text, gt_labels, pred_labels
        if self.model_type == "text_ner":
            example_lst = self.update_text_ner_example_lst(example_lst)
            pred_txt_lst = [line.split("\t")[0] for line in example_lst]
        else:
            pred_txt_lst = self.reorder(example_lst)
        assert len(example_lst) == len(pred_txt_lst)
        return example_lst, pred_txt_lst

    def update_text_ner_example_lst(self, example_lst):
        def make_distinct(tag_lst):
            """
            Make enities disticnt in a list
            For instance, when eval_asr == True
            input: [('PER', 'MARY'), ('LOC', "SAINT PAUL'S"), ('PER', 'KIRKLEATHAM'), ('PER', 'MARY')]
            output: [('PER', 'MARY', 1), ('LOC', "SAINT PAUL'S", 1), ('PER', 'KIRKLEATHAM', 1), ('PER', 'MARY', 2)]
            """
            tag2cnt, new_tag_lst = {}, []
            for tag_item in tag_lst:
                _ = tag2cnt.setdefault(tag_item, 0)
                tag2cnt[tag_item] += 1
                tag, wrd = tag_item
                new_tag_lst.append((tag, wrd, tag2cnt[tag_item]))
            return new_tag_lst

        def get_new_labels(old_label_lst, sent):
            new_label_lst = []
            for label, idx1, idx2 in old_label_lst:
                new_label_lst.append(
                    (label, " ".join(sent.split(" ")[idx1 : idx2 + 1]))
                )
            return make_distinct(new_label_lst)

        new_lst = []
        for item in example_lst:
            sent, gt_labels, pred_labels = item.split("\t")
            new_gt_labels = get_new_labels(ast.literal_eval(gt_labels), sent)
            new_pred_labels = get_new_labels(ast.literal_eval(pred_labels), sent)
            new_lst.append("\t".join([sent, str(new_gt_labels), str(new_pred_labels)]))
        return new_lst

    def classify_error(self, example_lst, pred_txt_lst):
        all_cnt_dct, all_example_dct = {}, {}
        all_sent_error_category_dct = (
            {}
        )  # {sent: {gt_item1: category_tag, gt_item2: category_tag}}
        mislabel_confusion_map = {}  # {(gt_tag, pred_tag): cnt}
        tot_num_gt_entities, tot_num_pred_entities = 0, 0
        for idx, example in enumerate(example_lst):
            pred_txt = pred_txt_lst[idx]
            (
                sent_error_category_dct,
                num_gt_entities,
                num_pred_entities,
            ) = self.process_sent(
                pred_txt, example, all_example_dct, all_cnt_dct, mislabel_confusion_map
            )
            tot_num_gt_entities += num_gt_entities
            tot_num_pred_entities += num_pred_entities
            all_sent_error_category_dct[
                example.split("\t")[0]
            ] = sent_error_category_dct
        print(f"# GT entities: {tot_num_gt_entities}")
        print(f"# Pred entities: {tot_num_pred_entities}")

        return (
            all_cnt_dct,
            all_example_dct,
            mislabel_confusion_map,
            all_sent_error_category_dct,
        )

    def process_sent(
        self, pred_txt, example, all_example_dct, all_cnt_dct, mislabel_confusion_map
    ):
        def update_dct(
            cnt_dct,
            example_dct,
            sent_error_category_dct,
            key,
            gt_entity_lst,
            pred_entity_lst,
            text,
        ):
            """
            Update the dictionary as per the classification category
            """
            _ = cnt_dct.setdefault(key, 0)
            _ = example_dct.setdefault(key, [])
            if "over_detection" not in key and "missed_detection" not in key:
                assert len(pred_entity_lst) == len(gt_entity_lst)
            if "missed_detection" in key:
                cnt_dct[key] += len(gt_entity_lst)
            else:
                cnt_dct[key] += len(pred_entity_lst)
            example_dct[key].append((text, gt_entity_lst, pred_entity_lst))
            for gt_item in gt_entity_lst:
                # if gt_item != -1: # cases of over-detection
                if "over_detection" not in key:
                    assert gt_item not in sent_error_category_dct
                    sent_error_category_dct[gt_item] = key

        def clean_lst(all_item_lst, item, tags_lst=None, phrases_lst=None):
            """
            Remove items that have already been accounted for
            """
            if isinstance(item, list):
                for ind_item in item:
                    all_item_lst.remove(ind_item)
            else:
                all_item_lst.remove(item)
                tags_lst.remove(item[0])
                phrases_lst.remove(item[1])

        def filter_stop_wrds(phrase):
            """
            Remove function words
            """
            out_str = []
            for wrd in phrase.split(" "):
                if wrd not in FUNC_WRDS:
                    out_str.append(wrd)
            return " ".join(out_str)

        def check_for_incomplete_detection(in_gt_phrase, item_lst):
            gt_phrase = filter_stop_wrds(in_gt_phrase)
            wrd_lst_1 = gt_phrase.split(" ")
            for item in item_lst:
                assert in_gt_phrase != item[1]  # not a case of mislabeling
                phrase = filter_stop_wrds(item[1])
                wrd_lst_2 = phrase.split(" ")
                if len(wrd_lst_1) > 0 or len(wrd_lst_2) > 0:
                    # if (len(set(wrd_lst_2) - set(wrd_lst_1)) - len(set(wrd_lst_2))) < 0 or \
                    # (len(set(wrd_lst_1) - set(wrd_lst_2)) - len(set(wrd_lst_1))) < 0:
                    if (len(set(wrd_lst_2) - set(wrd_lst_1)) - len(set(wrd_lst_2))) < 0:
                        return item, "over_gt"
                    elif (
                        len(set(wrd_lst_1) - set(wrd_lst_2)) - len(set(wrd_lst_1))
                    ) < 0:
                        return item, "in_gt"
            return False, ""

        def check_for_misspelling(in_gt_phrase, item_lst):
            gt_phrase = filter_stop_wrds(in_gt_phrase)
            frac = 0.4  # at least 60% of characters are a match
            for item in item_lst:
                assert in_gt_phrase != item[1]  # not a case of mislabeling
                phrase = filter_stop_wrds(item[1])
                thresh = frac * max(len(gt_phrase), len(phrase))
                diff_score = editdistance.eval(gt_phrase, phrase)
                if diff_score < thresh:
                    return item
            return False

        def update_global_dct(all_cnt_dct, all_example_dct, cnt_dct, example_dct):
            """
            Update the global dict after processing through each example
            """
            for key, cnt in cnt_dct.items():
                _ = all_cnt_dct.setdefault(key, 0)
                all_cnt_dct[key] += cnt

            for key, example_lst in example_dct.items():
                _ = all_example_dct.setdefault(key, [])
                gt_entity_lst, pred_entity_lst = [], []
                for text, gt_entities, pred_entities in example_lst:
                    gt_entity_lst.extend(gt_entities)
                    pred_entity_lst.extend(pred_entities)
                all_example_dct[key].append((text, gt_entity_lst, pred_entity_lst))

        sent_error_category_dct = {}  # {gt_item1: category_tag, gt_item2: category_tag}
        num_entities = 0
        cnt_dct, example_dct = {}, {}
        if len(example.split("\t")) == 3:
            text, gt_str, pred_str = example.split("\t")
            gt_lst = ast.literal_eval(gt_str)
            pred_lst = ast.literal_eval(pred_str)
            num_gt_entities = len(gt_lst)
            num_pred_entities = len(pred_lst)
            num_entities += len(gt_lst)
            if set(gt_lst) == set(pred_lst):
                update_dct(
                    cnt_dct,
                    example_dct,
                    sent_error_category_dct,
                    "correct",
                    gt_lst,
                    pred_lst,
                    text,
                )
            else:
                correct_predictions = list(set(gt_lst) - (set(gt_lst) - set(pred_lst)))
                update_dct(
                    cnt_dct,
                    example_dct,
                    sent_error_category_dct,
                    "correct",
                    correct_predictions,
                    correct_predictions,
                    text,
                )
                clean_lst(gt_lst, correct_predictions)
                clean_lst(pred_lst, correct_predictions)

                gt_phrases = [item[1] for item in gt_lst]
                gt_tags = [item[0] for item in gt_lst]
                unaccounted_pred_items = []
                for pred_item in pred_lst:
                    assert (
                        pred_item not in gt_lst
                    )  # verify that all "correct" ones are already accounted for
                    if pred_item[1] in gt_phrases:  # mislabeling
                        for item in gt_lst:
                            if item[1] == pred_item[1]:
                                gt_item = item
                                break
                        update_dct(
                            cnt_dct,
                            example_dct,
                            sent_error_category_dct,
                            "mislabel",
                            [gt_item],
                            [pred_item],
                            text,
                        )
                        sent_error_category_dct[gt_item] = "mislabel"
                        clean_lst(gt_lst, gt_item, gt_tags, gt_phrases)
                        mislabel_tuple = (gt_item[0], pred_item[0])
                        _ = mislabel_confusion_map.setdefault(mislabel_tuple, 0)
                        mislabel_confusion_map[mislabel_tuple] += 1
                    else:  # a case of misdetection or overdetection; handled separately
                        unaccounted_pred_items.append(pred_item)

                if (
                    len(unaccounted_pred_items) != 0 or len(gt_lst) != 0
                ):  # all misdetection cases
                    pred_phrases = [item[1] for item in unaccounted_pred_items]
                    pred_tags = [item[0] for item in unaccounted_pred_items]
                    for gt_item in gt_lst:
                        assert (
                            gt_item[1] not in pred_phrases
                        )  # verify that this is indeed a case of misdetection
                        gt_phrase = gt_item[1]
                        pred_item, classifier_str = check_for_incomplete_detection(
                            gt_phrase, unaccounted_pred_items
                        )
                        if (
                            pred_item
                        ):  # incomplete detection; checked using word overlap
                            if gt_item[1] not in pred_txt:
                                if pred_item[0] == gt_item[0]:
                                    update_dct(
                                        cnt_dct,
                                        example_dct,
                                        sent_error_category_dct,
                                        "partial_detect_correct_tag",
                                        [gt_item],
                                        [pred_item],
                                        text,
                                    )
                                else:
                                    update_dct(
                                        cnt_dct,
                                        example_dct,
                                        sent_error_category_dct,
                                        "partial_detect_incorrect_tag",
                                        [gt_item],
                                        [pred_item],
                                        text,
                                    )
                            else:
                                # if pred_item[0] == gt_item[0]:
                                #     update_dct(cnt_dct, example_dct, sent_error_category_dct, "partial_detect_correct_asr_correct_tag", [gt_item], [pred_item], text)
                                # else:
                                #     update_dct(cnt_dct, example_dct, sent_error_category_dct, "partial_detect_correct_asr_incorrect_tag", [gt_item], [pred_item], text)
                                update_dct(
                                    cnt_dct,
                                    example_dct,
                                    sent_error_category_dct,
                                    f"partial_detect_correct_asr_{classifier_str}",
                                    [gt_item],
                                    [pred_item],
                                    text,
                                )
                            clean_lst(
                                unaccounted_pred_items,
                                pred_item,
                                pred_tags,
                                pred_phrases,
                            )
                        else:  # misspelling; checked using threshold on editdistance
                            pred_item = check_for_misspelling(
                                gt_phrase, unaccounted_pred_items
                            )
                            if pred_item:
                                if pred_item[0] == gt_item[0]:
                                    update_dct(
                                        cnt_dct,
                                        example_dct,
                                        sent_error_category_dct,
                                        "misspelled_correct_tag",
                                        [gt_item],
                                        [pred_item],
                                        text,
                                    )
                                else:
                                    update_dct(
                                        cnt_dct,
                                        example_dct,
                                        sent_error_category_dct,
                                        "misspelled_incorrect_tag",
                                        [gt_item],
                                        [pred_item],
                                        text,
                                    )
                                clean_lst(
                                    unaccounted_pred_items,
                                    pred_item,
                                    pred_tags,
                                    pred_phrases,
                                )
                            else:  # complete misdetection
                                if gt_item[1] in pred_txt:
                                    # update_dct(cnt_dct, example_dct, sent_error_category_dct,
                                    #     "missed_detection_correctly_spelled", [gt_item], [-1], text)
                                    update_dct(
                                        cnt_dct,
                                        example_dct,
                                        sent_error_category_dct,
                                        "missed_detection_correctly_spelled",
                                        [gt_item],
                                        pred_lst,
                                        text,
                                    )
                                    mislabel_tuple = (gt_item[0], "null")
                                    _ = mislabel_confusion_map.setdefault(
                                        mislabel_tuple, 0
                                    )
                                    mislabel_confusion_map[mislabel_tuple] += 1
                                else:
                                    # update_dct(cnt_dct, example_dct, sent_error_category_dct,
                                    #     "missed_detection_misspelled", [gt_item], [-1], text)
                                    update_dct(
                                        cnt_dct,
                                        example_dct,
                                        sent_error_category_dct,
                                        "missed_detection_misspelled",
                                        [gt_item],
                                        pred_lst,
                                        text,
                                    )

                if len(unaccounted_pred_items) != 0:  # all over-detection cases
                    items_with_misspelling = [
                        item for item in unaccounted_pred_items if item[1] not in text
                    ]
                    items_correctly_spelled = [
                        item for item in unaccounted_pred_items if item[1] in text
                    ]
                    assert len(unaccounted_pred_items) == (
                        len(items_with_misspelling) + len(items_correctly_spelled)
                    )
                    if len(items_with_misspelling) > 0:
                        # update_dct(cnt_dct, example_dct, sent_error_category_dct,
                        #     "over_detection_misspelled", [-1]*len(items_with_misspelling), items_with_misspelling, text)
                        update_dct(
                            cnt_dct,
                            example_dct,
                            sent_error_category_dct,
                            "over_detection_misspelled",
                            gt_lst,
                            items_with_misspelling,
                            text,
                        )
                    if len(items_correctly_spelled) > 0:
                        # update_dct(cnt_dct, example_dct, sent_error_category_dct, "over_detection_correctly_spelled"
                        #     [-1]*len(items_correctly_spelled), items_correctly_spelled, text)
                        update_dct(
                            cnt_dct,
                            example_dct,
                            sent_error_category_dct,
                            "over_detection_correctly_spelled",
                            gt_lst,
                            items_correctly_spelled,
                            text,
                        )
                        for item in items_correctly_spelled:
                            mislabel_tuple = ("null", item[0])
                            _ = mislabel_confusion_map.setdefault(mislabel_tuple, 0)
                            mislabel_confusion_map[mislabel_tuple] += 1

            est_num_gt_entities, est_num_pred_entities = 0, 0
            for key in self.gt_category_keys:
                if key in cnt_dct:
                    est_num_gt_entities += cnt_dct[key]
            for key in self.pred_category_keys:
                if key in cnt_dct:
                    est_num_pred_entities += cnt_dct[key]
            # assert num_gt_entities == est_num_gt_entities
            # assert num_pred_entities == est_num_pred_entities
            update_global_dct(all_cnt_dct, all_example_dct, cnt_dct, example_dct)
        else:
            num_gt_entities, num_pred_entities = 0, 0
        return sent_error_category_dct, num_gt_entities, num_pred_entities

    def convert_dct_to_cmat(self, cnt_dct):
        """
        Convert tag_pair counts dictionary to matrix
        """
        y_labels = list(set([item for _, item in cnt_dct]))
        x_labels = list(set([item for item, _ in cnt_dct]))
        x_labels.sort()
        y_labels.sort()

        x_label_to_idx, y_label_to_idx = {}, {}
        for idx, xlabel in enumerate(x_labels):
            x_label_to_idx[xlabel] = idx
        for idx, ylabel in enumerate(y_labels):
            y_label_to_idx[ylabel] = idx

        cmap = np.zeros([len(x_labels), len(y_labels)])
        for key, value in cnt_dct.items():
            cmap[x_label_to_idx[key[0]], y_label_to_idx[key[1]]] = value

        return cmap, x_labels, y_labels

    def vis_mislabel_cmap(self, mislabel_dct, write_dir):
        """
        Visualize confusion map
        """
        write_dir = os.path.join(write_dir, "plots")
        os.makedirs(write_dir, exist_ok=True)

        cmap, x_labels, y_labels = self.convert_dct_to_cmat(mislabel_dct)
        cmap_df = pd.DataFrame(cmap, index=x_labels, columns=y_labels)
        hmap = sn.heatmap(cmap_df, cmap="Spectral", xticklabels=True, yticklabels=True)
        hmap.set_title("Mislabel confusion maps")
        save_fn = os.path.join(write_dir, "mislabel-cmap.png")
        hmap.figure.savefig(save_fn, format="png", dpi=150)
        plt.close()

    def save_as_tsv(self, example_dct, write_dir):
        write_dir = os.path.join(write_dir, "examples")
        os.makedirs(write_dir, exist_ok=True)

        for key, value in example_dct.items():
            write_lst = []
            for sent, gt, pred in value:
                gt = str([gt_item[:2] for gt_item in gt])
                pred = str([pred_item[:2] for pred_item in pred])
                write_lst.append("\t".join([sent, gt, pred]))
            write_to_file("\n".join(write_lst), os.path.join(write_dir, f"{key}.tsv"))
        save_dct(os.path.join(write_dir, "all-examples.json"), example_dct)

    def get_error_dist(self):
        example_lst, pred_txt_lst = self.read_files()
        write_dir = os.path.join(
            "save", "error_distribution", os.path.basename(self.model_dir)
        )
        os.makedirs(write_dir, exist_ok=True)
        (
            cnt_dct,
            example_dct,
            mislabel_dct,
            sent_error_category_dct,
        ) = self.classify_error(example_lst, pred_txt_lst)
        self.save_as_tsv(example_dct, write_dir)
        save_dct(os.path.join(write_dir, "cnt_dct.json"), cnt_dct)
        save_dct(os.path.join(write_dir, "mislabel_dct.pkl"), mislabel_dct)
        self.vis_mislabel_cmap(mislabel_dct, write_dir)
        print(
            "Error analysis outputs for model %s saved at %s"
            % (os.path.basename(self.model_dir), write_dir)
        )


if __name__ == "__main__":
    fire.Fire(ClassifyErrors)
