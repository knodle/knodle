import os
from typing import Dict, Tuple, List

import numpy as np
import json
import math

from keras.utils import to_categorical, pad_sequences
from knodle.transformation.majority import seq_input_to_majority_vote_input

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"


def reassign_unk_pad(train_sents, dev_sents, test_sents, train_lfs, dev_lfs, test_lfs, word_to_id, lf_to_id,
                     other_class_label: str = "O"):
    id_to_word = {word_to_id[word]: word for word in word_to_id}
    id_to_lf = {lf_to_id[lf]: lf for lf in lf_to_id}

    for i, j in [(word_to_id[PAD_TOKEN], 0), (word_to_id[UNK_TOKEN], 1)]:
        train_sents = switch_idx(train_sents, i, j)
        dev_sents = switch_idx(dev_sents, i, j)
        test_sents = switch_idx(test_sents, i, j)

    train_lfs = switch_idx(train_lfs, lf_to_id[other_class_label], 0)
    dev_lfs = switch_idx(dev_lfs, lf_to_id[other_class_label], 0)
    test_lfs = switch_idx(test_lfs, lf_to_id[other_class_label], 0)
    word_to_id[id_to_word[0]] = word_to_id[PAD_TOKEN]
    word_to_id[id_to_word[1]] = word_to_id[UNK_TOKEN]
    lf_to_id[id_to_lf[0]] = lf_to_id[other_class_label]
    lf_to_id[other_class_label] = 0
    word_to_id[PAD_TOKEN] = 0
    word_to_id[UNK_TOKEN] = 1
    id_to_word = {word_to_id[word]: word for word in word_to_id}
    id_to_lf = {lf_to_id[label]: label for label in lf_to_id}
    return train_lfs, dev_lfs, test_lfs, word_to_id, lf_to_id, id_to_word, id_to_lf


def switch_idx(sents, i, j):
    D = {i: j, j: i}
    return [[D.get(word, word) for word in sent] for sent in sents]


def do_padding(sequences, max_length: int = 20):
    return pad_sequences(sequences, maxlen=max_length)


def get_train_dev_test_data(data: Dict) -> Tuple[List, List, List, List, List, List]:
    train_dev_sents = data["train_sents"]  # list of lists
    train_dev_lfs = data["train_labels"]  # list of lists

    num_train = math.floor(0.8 * len(train_dev_sents))
    train_sents = train_dev_sents[:num_train]
    train_lfs = train_dev_lfs[:num_train]
    dev_sents = train_dev_sents[num_train:]
    dev_lfs = train_dev_lfs[num_train:]
    test_sents = data["test_sents"]
    test_lfs = data["test_labels"]

    return train_sents, train_lfs, dev_sents, dev_lfs, test_sents, test_lfs


def read_atis(path_to_atis: str = None, other_class_label: str = "O", other_class_id: int = 0):
    if path_to_atis is None:
        path_to_atis = os.path.join(__location__, "atis.json")

    with open(path_to_atis, "rb") as input_file:
        data = json.load(input_file)

    # get the samples and LFs
    train_sents, train_lfs, dev_sents, dev_lfs, test_sents, test_lfs = get_train_dev_test_data(data)

    word_to_id = data["vocab"]
    lf_to_id = data["label_dict"]
    num_lfs = len(lf_to_id)

    #  We reorganize the data so that "<PAD>" has id 0, "<UNK>" has id 1, and the label 'O' has id 0 (and all words
    #  are still represented). Make sure to also change word_to_id and label_to_id.
    train_lfs, dev_lfs, test_lfs, word_to_id, lf_to_id, id_to_word, id_to_lf = reassign_unk_pad(
        train_sents, dev_sents, test_sents, train_lfs, dev_lfs, test_lfs, word_to_id, lf_to_id
    )

    # pad sentences
    train_sents_padded = do_padding(train_sents, max_length=20)
    dev_sents_padded = do_padding(dev_sents, max_length=20)
    test_sents_padded = do_padding(test_sents, max_length=20)

    # extract pure labels
    new_labels = [oldlabel.split(".")[0].split("-")[-1] for oldlabel in lf_to_id.keys()]
    lf_to_newlabel = dict(zip(lf_to_id.keys(), new_labels))
    del lf_to_newlabel["O"]

    # 1. use original labels, without "O" to represent Z-matrix
    # Dimensions of XXX_lfs_padded: #samples x #tokens in each sample x #LFs
    train_lfs_padded = to_categorical(do_padding(train_lfs, max_length=20), num_lfs)[:, :, 1:]
    dev_lfs_padded = to_categorical(do_padding(dev_lfs, max_length=20), num_lfs)[:, :, 1:]
    test_lfs_padded = to_categorical(do_padding(test_lfs, max_length=20), num_lfs)[:, :, 1:]

    # create dict with cleaned labels to label ids
    new_labels = list(set(lf_to_newlabel.values()))  # cut the BIO schema
    newlabel_to_id = {**{other_class_label: other_class_id},
                      **{new_labels[i]: i + 1 for i in range(0, len(new_labels))}}

    lfid_to_nlid = dict()
    for lf_id in id_to_lf:
        if lf_id == 0:
            continue
        lf_name = id_to_lf[lf_id]
        newlabel_name = lf_to_newlabel[lf_name]
        nl_id = newlabel_to_id[newlabel_name]
        # "O" is not a lf in the 1-hot encoding of the original labels
        lfid_to_nlid[lf_id - 1] = nl_id

    t_matrix = np.zeros((len(lfid_to_nlid), len(newlabel_to_id)))
    for lfid in lfid_to_nlid:
        t_matrix[lfid, lfid_to_nlid[lfid]] = 1

    id_to_label = {i: l for l, i in newlabel_to_id.items()}

    dev_labels_padded = seq_input_to_majority_vote_input(dev_lfs_padded, t_matrix).argmax(axis=2)
    # test_labels = majority_vote(test_labels_padded, t_matrix).argmax(axis=2)
    return train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, \
           t_matrix, id_to_word, id_to_lf, id_to_label
