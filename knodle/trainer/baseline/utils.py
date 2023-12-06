from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
        self.data_size = len(self.tokens)

    def __getitem__(self, i):
        """ Return one training/test sample at once """
        return self.tokens[i], self.labels[i]

    def __len__(self):
        return self.data_size


def accuracy_padded(predicted, gold, mask):
    # Expects label ids, NOT 1-hot encodings
    # predicted $ gold $ mask: batch_size x seqlength
    return sum(sum((predicted == gold) * mask)) / sum(sum(mask))



