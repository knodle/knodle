from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
        self.data_size = len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]

    def __len__(self):
        return self.data_size


def accuracy_padded(predicted, gold, mask):
    # Expects label ids, NOT 1-hot encodings
    return sum(sum((predicted == gold) * mask)) / sum(sum(mask))



