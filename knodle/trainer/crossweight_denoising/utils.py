import numpy as np


def get_labels(rule_matches_z: np.ndarray, rule_assignments_t: np.ndarray) -> np.ndarray:
    """ Calculates sample labels basing on z and t matrices """

    assert rule_matches_z.shape[1] == rule_assignments_t.shape[0], "Check matrices dimensionality!"

    one_hot_labels = rule_matches_z.dot(rule_assignments_t)  # calculate labels
    one_hot_labels[one_hot_labels > 0] = 1
    labels = [np.where(r == 1)[0][0] for r in one_hot_labels]
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
            parts = line.strip().split(' ')
            word = parts[0]
            if word not in word_to_id:
                emb = [float(v) for v in parts[1:]]
                matrix[nextword_id] = emb
                word_to_id[word] = nextword_id
                nextword_id += 1
    return word_to_id, matrix


def add_padding(tokens: list, maxlen: int) -> list:
    """ Provide padding of the encoded tokens to the maxlen; if length of tokens > maxlen, reduce it to maxlen """
    padded_tokens = [0] * maxlen
    for i in range(0, min(len(tokens), maxlen)):
        padded_tokens[i] = tokens[i]
    return padded_tokens


def get_embedding_matrix(pretrained_embedding_file: str) -> np.ndarray:
    """ Return matrix with pretrained glove embeddings"""
    with open(pretrained_embedding_file, encoding="UTF-8") as in_file:
        emb_matrix_size = in_file.readline().strip().split(" ")
        embeddings = []
        for line in in_file:
            parts = line.strip().split(' ')
            embeddings.append([float(v) for v in parts[1:]])
        emb_matrix = np.array(embeddings)
        assert emb_matrix.shape[0] == int(emb_matrix_size[0]) and emb_matrix.shape[1] == int(emb_matrix_size[1])
    return emb_matrix