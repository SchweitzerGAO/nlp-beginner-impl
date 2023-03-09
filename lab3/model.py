import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import load_train_data
from public.misc import TokenEmbedding


def local_inference(A_bar, B_bar, E_row, E_col):
    A_len = A_bar.shape[1]
    dim_vec = A_bar.shape[2]
    B_len = B_bar.shape[1]

    def sum_local_inference(idx, dim):
        if dim == 1:
            a = E_row[:, idx, :]
        else:
            a = E_col[:, :, idx]
        b = a.unsqueeze(2).repeat_interleave(dim_vec, dim=2)
        c = b * B_bar if dim == 1 else b * A_bar
        return torch.sum(c, dim=1)

    A_tilde = torch.stack([sum_local_inference(i, dim=1) for i in range(A_len)]).permute(1, 0, 2)
    B_tilde = torch.stack([sum_local_inference(j, dim=2) for j in range(B_len)]).permute(1, 0, 2)
    return A_tilde, B_tilde


def enhancement(bar, tilde):
    sub = bar - tilde
    mul = bar * tilde
    M = torch.cat((bar, tilde, sub, mul), dim=2)
    return M


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.bi_lstm = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               batch_first=True,
                               bidirectional=True)

    def forward(self, A, B):
        # input encoding
        embedded_A = self.embed(A.long())
        A_bar, _ = self.bi_lstm(embedded_A)
        embedded_B = self.embed(B.long())
        B_bar, _ = self.bi_lstm(embedded_B)

        # local inference (token2token attention)
        A_bar_T = A_bar.permute(0, 2, 1)
        E = torch.bmm(B_bar, A_bar_T).permute(0, 2, 1)
        E_row_softmax = F.softmax(E, dim=2)
        E_col_softmax = F.softmax(E, dim=1)

        A_tilde, B_tilde = local_inference(A_bar, B_bar, E_row_softmax, E_col_softmax)

        # enhancement
        M_A = enhancement(A_bar, A_tilde)
        M_B = enhancement(B_bar, B_tilde)

        return M_A, M_B


class Decoder(nn.Module):
    def __init__(self, input_size, length, hidden_size_lstm, hidden_size_dense, output_size, dropout=0.5):
        super().__init__()
        self.bi_lstm = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size_lstm,
                               batch_first=True,
                               bidirectional=True)
        self.ave_pool = nn.AvgPool1d(kernel_size=length)
        self.max_pool = nn.MaxPool1d(kernel_size=length)
        self.fc1 = nn.Linear(input_size, hidden_size_dense)
        self.fc2 = nn.Linear(hidden_size_dense, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, M_A, M_B):
        V_A, _ = self.bi_lstm(M_A)
        V_B, _ = self.bi_lstm(M_B)
        V_A = V_A.permute(0, 2, 1)
        V_B = V_B.permute(0, 2, 1)
        A_ave = self.ave_pool(V_A).squeeze(2)
        B_ave = self.ave_pool(V_B).squeeze(2)
        A_max = self.max_pool(V_A).squeeze(2)
        B_max = self.max_pool(V_B).squeeze(2)
        V = torch.cat((A_ave, A_max, B_ave, B_max), dim=1)
        out1 = self.fc1(self.dropout(V))
        out = self.fc2(self.dropout(out1))
        return out


class ESIM(nn.Module):
    def __init__(self, vocab, embed_size, length, hidden_size_lstm, hidden_size_dense, output_size, dropout=0.5):
        super().__init__()
        self.encoder = Encoder(len(vocab), embed_size, hidden_size_lstm)
        self.decoder = Decoder(hidden_size_lstm * 2 * 4, length, hidden_size_lstm,
                               hidden_size_dense, output_size, dropout=dropout)
        wv = TokenEmbedding('../public/glove_6B_100d.pkl')
        weight_embed = wv[vocab.idx_to_token]
        self.encoder.embed.weight.data.copy_(weight_embed)
        self.encoder.embed.weight.requires_grad = False

    def forward(self, a, b):
        m_a, m_b = self.encoder(a, b)
        out = self.decoder(m_a, m_b)
        return out


if __name__ == '__main__':
    embed_size = 100
    hidden_size = 128
    train_loader, vocab = load_train_data(test=True)
    net = ESIM(vocab, embed_size=100, length=50, hidden_size_lstm=128, hidden_size_dense=128, output_size=3)
    net.eval()
    for (A, B), y in train_loader:
        y_hat = net(A, B)
        y_softmax = F.softmax(y_hat, dim=1)
        pass
