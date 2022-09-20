from torch import nn


# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/blob/master/train.py
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, padding_idx=0):
        super(LSTMModel, self).__init__()
        self.tagset_size = tagset_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.emission = nn.Linear(hidden_dim, tagset_size)

    def forward(self, token_idxs):
        embeds = self.embeds(token_idxs)
        hidden, _ = self.lstm(embeds)
        pred = self.emission(hidden)
        return pred
