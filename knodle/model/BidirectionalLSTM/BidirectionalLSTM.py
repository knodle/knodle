import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
# from lstm_huggingface import LSTMHardSigmoid


class BidirectionalLSTM(nn.Module):

    def __init__(self, word_input_dim, word_output_dim, word_embedding_matrix, num_classes, size_factor=200):
        super(BidirectionalLSTM, self).__init__()

        self.word_input_dim = word_input_dim            # 66449
        self.word_output_dim = word_output_dim      # 300
        self.word_embedding_matrix = word_embedding_matrix      # (66449, 300)
        # self.num_types = num_types      # 20
        self.size_factor = size_factor          # 300
        self.num_classes = num_classes      # 42

        self.word_embedding = nn.Embedding(word_input_dim, word_output_dim, padding_idx=0)
        self.word_embedding.weight = nn.Parameter(torch.tensor(word_embedding_matrix, dtype=torch.float32))
        self.word_embedding.weight.requires_grad = False

        # self.type_embedding = nn.Embedding(num_types, 10)
        # self.type_embedding.weight = nn.init.xavier_normal_(self.type_embedding.weight)

        self.type_linear = nn.Linear(20, size_factor*2)

        self.td_dense = nn.Linear(word_output_dim, size_factor)
        self.biLSTM = nn.LSTM(size_factor, size_factor, bidirectional=True, batch_first=True)

        self.predict = nn.Linear(size_factor*2, num_classes)
        self.init_weights()

    def forward(self,
                x,
                # y,
                # lens,
                ):
        word_embeddings = self.word_embedding(x)

        td_dense = self.td_dense(word_embeddings)
        x_packed = pack_padded_sequence(td_dense,
                                        # lens,
                                        batch_first=True, enforce_sorted=False)

        biLSTM, (h_n, c_n) = self.biLSTM(x_packed)
        self.biLSTM.flatten_parameters()   ##
        biLSTM, _ = pad_packed_sequence(biLSTM, batch_first=True)

        ###
        final_state = h_n.view(1, 2, x.shape[0], self.size_factor)[-1]
        h_1, h_2 = final_state[0], final_state[1]  # forward & backward pass
        concat = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # type embeddings + pooling
        # type_embeddings = self.type_embedding(y)
        # subj_type = type_embeddings[:, 0, :]
        # obj_type = type_embeddings[:, 1, :]
        # type_vecs = torch.cat((subj_type, obj_type), dim=1)
        # type_vec_of_same_length_as_lstm = self.type_linear(type_vecs)
        # global_repr = concat.add(type_vec_of_same_length_as_lstm)

        # final = self.predict(global_repr)
        final = self.predict(concat)

        return final

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        torch.manual_seed(12345)
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        # nn.init.uniform_(self.type_embedding.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)



