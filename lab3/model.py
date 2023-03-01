import torch
import torch.nn as nn
from preprocess import load_train_data
from public.misc import TokenEmbedding
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.bi_lstm = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               batch_first=True,
                               bidirectional=True)

    def forward(self, A, B):
        A_len = A.shape[1]
        B_len = B.shape[1]
        embedded_A = self.embed(A.long())
        A_bar, _ = self.bi_lstm(embedded_A)
        embedded_B = self.embed(B.long())
        B_bar, _ = self.bi_lstm(embedded_B)
        A_bar_T = A_bar.permute(0, 2, 1)
        E = torch.bmm(B_bar, A_bar_T).permute(0, 2, 1)
        E_row_softmax = F.softmax(E, dim=2)
        E_col_softmax = F.softmax(E, dim=1)

        A_tilde = None
        B_tilde = None


class Decoder(nn.Module):
    pass


if __name__ == '__main__':
    embed_size = 100
    hidden_size = 128
    train_loader, vocab = load_train_data(test=True)
    wv = TokenEmbedding('../public/glove_6B_100d.pkl')
    weight_embed = wv[vocab.idx_to_token]
    encoder = Encoder(len(vocab), embed_size, hidden_size)
    encoder.embed.weight.data.copy_(weight_embed)
    encoder.embed.weight.requires_grad = False
    for (A, B), y in train_loader:
        A_tilde, B_tilde = encoder(A, B)
        pass
