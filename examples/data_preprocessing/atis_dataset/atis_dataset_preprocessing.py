import os
import numpy as np
import json
import math

from keras.utils import to_categorical, pad_sequences
from knodle.transformation.majority import seq_input_to_majority_vote_input

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
MAX_LENGTH = 20


def switch_idx(sents, i, j):
    D = {i: j, j: i}
    return [[D.get(word, word) for word in sent] for sent in sents]


def do_padding(sequences, length=MAX_LENGTH):
    return pad_sequences(sequences, maxlen=length)


def read_atis(atis_fn: str = "atis.json"):
    with open(os.path.join(__location__, atis_fn), "rb") as input_file:
        data = json.load(input_file)

    train_dev_sents = data["train_sents"]  # list of lists
    train_dev_lfs = data["train_labels"]  # list of lists
    num_train = math.floor(0.8 * len(train_dev_sents))
    train_sents = train_dev_sents[:num_train]
    train_lfs = train_dev_lfs[:num_train]
    dev_sents = train_dev_sents[num_train:]
    dev_lfs = train_dev_lfs[num_train:]
    test_sents = data["test_sents"]
    test_lfs = data["test_labels"]
    word_to_id = data["vocab"]
    lf_to_id = data["label_dict"]
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    NUM_LFs = len(lf_to_id)

    #  We reorganize the data so that "<PAD>" has id 0, "<UNK>" has id 1, and the label 'O' has id 0 (and all words
    #  are still represented). Make sure to also change word_to_id and label_to_id.
    id_to_word = {word_to_id[word]: word for word in word_to_id}
    id_to_lf = {lf_to_id[lf]: lf for lf in lf_to_id}

    for i, j in [(word_to_id[PAD_TOKEN], 0), (word_to_id[UNK_TOKEN], 1)]:
        train_sents = switch_idx(train_sents, i, j)
        dev_sents = switch_idx(dev_sents, i, j)
        test_sents = switch_idx(test_sents, i, j)

    train_lfs = switch_idx(train_lfs, lf_to_id["O"], 0)
    dev_lfs = switch_idx(dev_lfs, lf_to_id["O"], 0)
    test_lfs = switch_idx(test_lfs, lf_to_id["O"], 0)
    word_to_id[id_to_word[0]] = word_to_id[PAD_TOKEN]
    word_to_id[id_to_word[1]] = word_to_id[UNK_TOKEN]
    lf_to_id[id_to_lf[0]] = lf_to_id["O"]
    lf_to_id["O"] = 0
    word_to_id[PAD_TOKEN] = 0
    word_to_id[UNK_TOKEN] = 1
    id_to_word = {word_to_id[word]: word for word in word_to_id}
    id_to_lf = {lf_to_id[label]: label for label in lf_to_id}

    train_sents_padded = do_padding(train_sents)
    dev_sents_padded = do_padding(dev_sents)
    test_sents_padded = do_padding(test_sents)

    lf_to_newlabel = dict()
    for oldlabel in lf_to_id.keys():
        newlabel = oldlabel.split(".")[0].split("-")[-1]
        lf_to_newlabel[oldlabel] = newlabel
    del lf_to_newlabel["O"]

    # 1. use original labels, without "O" to represent Z-matrix
    train_lfs_padded = to_categorical(do_padding(train_lfs), NUM_LFs)[:, :, 1:]
    dev_lfs_padded = to_categorical(do_padding(dev_lfs), NUM_LFs)[:, :, 1:]
    test_lfs_padded = to_categorical(do_padding(test_lfs), NUM_LFs)[:, :, 1:]

    newlabel_to_id = dict()
    OTHER_ID = 0
    OTHER_LABEL = "O"
    newlabel_to_id[OTHER_LABEL] = OTHER_ID
    for i, nl in enumerate(set(lf_to_newlabel.values())):
        newlabel_to_id[nl] = i + 1

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
