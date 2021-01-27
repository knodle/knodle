import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_labels(
        rule_matches_z: np.ndarray, rule_assignments_t: np.ndarray
) -> np.ndarray:
    """ Calculates sample labels basing on z and t matrices. If several patterns matched, select one randomly """

    if rule_matches_z.shape[1] != rule_assignments_t.shape[0]:
        raise ValueError("Dimensions mismatch!")

    one_hot_labels = rule_matches_z.dot(rule_assignments_t)
    one_hot_labels[one_hot_labels > 0] = 1
    labels = [np.random.choice(np.where(r == 1)[0], 1)[0] for r in one_hot_labels]
    return np.stack(labels, axis=0)


def vocab_and_vectors(filename: str, special_tokens: list) -> (dict, dict, np.ndarray):
    """special tokens have all-zero word vectors"""
    with open(filename, encoding="UTF-8") as in_file:
        parts = in_file.readline().strip().split(" ")
        num_vecs = int(parts[0]) + len(special_tokens)  # + 1
        dim = int(parts[1])

        matrix = np.zeros((num_vecs, dim))
        word_to_id = dict()

        nextword_id = 0
        for token in special_tokens:
            word_to_id[token] = nextword_id
            nextword_id += 1

        for line in in_file:
            parts = line.strip().split(" ")
            word = parts[0]
            if word not in word_to_id:
                emb = [float(v) for v in parts[1:]]
                matrix[nextword_id] = emb
                word_to_id[word] = nextword_id
                nextword_id += 1
    return word_to_id, matrix


def get_embedding_matrix(pretrained_embedding_file: str) -> np.ndarray:
    """ Return matrix with pretrained glove embeddings"""
    with open(pretrained_embedding_file, encoding="UTF-8") as in_file:
        emb_matrix_size = in_file.readline().strip().split(" ")
        embeddings = []
        for line in in_file:
            parts = line.strip().split(" ")
            embeddings.append([float(v) for v in parts[1:]])
        emb_matrix = np.array(embeddings)
        assert emb_matrix.shape[0] == int(emb_matrix_size[0]) and emb_matrix.shape[
            1
        ] == int(emb_matrix_size[1])
    return emb_matrix


def set_seed(seed: int):
    """ Fix seed for all shuffle processes in order to get the reproducible result """
    np.random.seed(np.array(seed, dtype="int64"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_device(enable_cuda: bool):
    """ Set where the calculations will be done (cpu or cuda) depending on whether the cuda is available and chosen """
    if enable_cuda and torch.cuda.is_available():
        logger.info("Using GPU")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def check_splitting(
        tst_samples: torch.Tensor,
        tst_labels: np.ndarray,
        tst_idx: np.ndarray,
        samples: torch.Tensor,
        labels: np.ndarray,
) -> None:
    """ Custom function to check that the splitting into train and test sets fro CrossWeigh was done correctly"""

    rnd_tst = np.random.randint(0, tst_samples.shape[0])  # take some random index
    tst_sample = tst_samples[rnd_tst, :]
    tst_idx = tst_idx[rnd_tst]
    tst_label = tst_labels[rnd_tst, :]

    if not torch.equal(tst_sample, samples[tst_idx, :]):
        raise RuntimeError(
            "The splitting of original training set into cw train and test sets have been done "
            "incorrectly! A sample does not correspond to one in original dataset"
        )

    if not np.array_equal(tst_label, labels[tst_idx, :]):
        raise RuntimeError(
            "The splitting of original training set into cw train and test sets have been done "
            "incorrectly! A sample label does not correspond to one in original dataset"
        )


def return_unique(where_to_find: np.ndarray, what_to_find: np.ndarray) -> np.ndarray:
    """ Checks intersections between the 1st and the 2nd arrays and return unique values of the 1st array """
    intersections = np.intersect1d(where_to_find, what_to_find, return_indices=True)[
        1
    ].tolist()
    return np.delete(where_to_find, intersections)


def make_plot(
        value_1: list,
        value_2: list,
        value_3: list,
        value_4: list,
        label_1: str,
        label_2: str,
        label_3: str,
        label_4: str,
):
    """ The function creates a plot of 4 curves and displays it"""
    plt.plot(value_1, "g", label=label_1)
    plt.plot(value_2, "r", label=label_2)
    plt.plot(value_3, "b", label=label_3)
    plt.plot(value_4, "y", label=label_4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.show()
