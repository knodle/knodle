from examples.labeler.chexpert.constants.constants import *


def t_matrix_fct():
    """Create T-matrix from rules (mentions)."""

    mentions = pd.concat([pd.read_csv(os.path.join(MENTION_DATA_DIR, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in FILES], ignore_index=True)

    T_matrix = pd.DataFrame(data=mentions).iloc[:, 1].str.get_dummies().to_numpy()

    return T_matrix


def z_matrix_fct():
    """Create T-matrix from rules (mentions)."""

    mentions = pd.concat([pd.read_csv(os.path.join(MENTION_DATA_DIR, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in FILES], ignore_index=True)
    n_rules = len(mentions[0])

    reports = pd.read_csv(REPORTS_PATH,
                          header=None,
                          names=[REPORTS])[REPORTS].tolist()
    n_samples = len(reports)

    Z_matrix = np.zeros((n_samples, n_rules))  # , dtype=int

    return Z_matrix


def get_rule_idx(phrase):
    """Given phrase, outputs number of rule."""

    mentions = pd.concat([pd.read_csv(os.path.join(MENTION_DATA_DIR, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in FILES], ignore_index=True)

    index = mentions.index[mentions[0] == phrase]

    return index


def transform(text_file):
    """Transform file of words to patterns which are compatible with ngrex."""

    file = open(text_file, "r+")

    new_file = []

    for line in file:
        lemmatized1 = "{} < {} {lemma:/" + str(line).rstrip() + "/}"
        new_file.append(lemmatized1)

        lemmatized2 = "{} > {} {lemma:/" + str(line).rstrip() + "/}"
        new_file.append(lemmatized2)

    with open(text_file, "w+") as file:
        for i in new_file:
            file.write(i + "\n")
