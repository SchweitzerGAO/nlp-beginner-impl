import torch
import torch.nn as nn

from preprocess import load_train_data
from public.misc import PretrainedEmbedding


class BiLSTMEmbed(nn.Module):
    def __init__(self, word_length, hidden_size, sent_vocab, embed_size=100):
        super().__init__()
        self.bi_lstm = nn.LSTM(word_length, hidden_size, batch_first=True, bidirectional=True)
        self.embed = nn.Embedding(len(sent_vocab), embed_size)
        wv = PretrainedEmbedding('../public/glove_6B_100d.pkl')
        weight_embed = wv[sent_vocab.idx_to_token]
        self.embed.weight.data.copy_(weight_embed)
        self.embed.weight.requires_grad = False

    def forward(self,C,S):




if __name__ == '__main__':
    train_loader, vocabs = load_train_data()
