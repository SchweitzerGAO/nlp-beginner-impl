import torch
import torch.nn as nn

from preprocess import load_train_data
from public.misc import PretrainedEmbedding


def argmax(data):
    _, max_idx = torch.max(data)
    return max_idx.item()


def log_sum_exp(data):
    max_score = data[0, argmax(data)]
    max_score_board = max_score.view(1, -1).expand(1, data.size(1))
    return max_score + torch.log(torch.sum(torch.exp(data - max_score_board)))


class BiLSTMEmbed(nn.Module):
    def __init__(self, word_length, sent_vocab, embed_size=100):
        super().__init__()
        self.bi_lstm = nn.LSTM(word_length, word_length // 2, batch_first=True, bidirectional=True)
        self.pretrained_embed = nn.Embedding(len(sent_vocab), embed_size)
        self.embed_size = embed_size + (word_length // 2) * 2

    def forward(self, C, S):
        output, _ = self.bi_lstm(C)
        pretrained = self.pretrained_embed(S)
        embedded = torch.cat((pretrained, output), dim=2)
        return embedded


class CNNEmbed(nn.Module):
    def __init__(self, char_vocab, sent_vocab, word_length, sent_length, window_size=3, embed_size=100):
        super().__init__()
        self.char_embed = nn.Embedding(len(char_vocab), word_length)
        self.dropout = nn.Dropout(0.5)
        self.cnn = nn.Conv2d(sent_length, sent_length, kernel_size=window_size)
        out_length = word_length - window_size + 1
        self.max_pool = nn.MaxPool1d(out_length)
        self.pretrained_embed = nn.Embedding(len(sent_vocab), embed_size)
        self.embed_size = embed_size + out_length

    def forward(self, C, S):
        output = self.char_embed(C.long())
        output = self.cnn(self.dropout(output))
        pool_list = []
        for i in range(output.size(2)):
            pool_list.append(self.max_pool(output[:, :, i, :]))
        output = torch.stack(pool_list, dim=2).squeeze(3)
        pretrained = self.pretrained_embed(S.long())
        embedded = torch.cat((pretrained, output), dim=2)
        return embedded


class Encoder(nn.Module):
    def __init__(self, vocabs, word_length, sent_length, hidden_size, char_embed='lstm', window_size=3,
                 embed_size=100):
        super().__init__()
        if char_embed == 'lstm':
            self.embed = BiLSTMEmbed(word_length, vocabs[1], embed_size)
        elif char_embed == 'cnn':
            self.embed = CNNEmbed(vocabs[0], vocabs[1], word_length, sent_length, window_size, embed_size)
        else:
            raise NotImplementedError
        wv = PretrainedEmbedding('../public/glove_6B_100d.pkl')
        weight_embed = wv[vocabs[1].idx_to_token]
        self.embed.pretrained_embed.weight.data.copy_(weight_embed)
        self.embed.pretrained_embed.weight.requires_grad = False
        self.bi_lstm_enc = nn.LSTM(self.embed.embed_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_size, len(vocabs[2]))

    def forward(self, C, S):
        embed = self.embed(C, S)
        out, _ = self.bi_lstm_enc(embed)
        out = self.hidden2tag(out)
        return out


class CRFDecoder(nn.Module):
    def __init__(self, label_vocab):
        super().__init__()
        self.labels = label_vocab
        # transition[i,j] is the score of transition from i to j (different to the implementation in
        # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py)
        self.transition = nn.Parameter(torch.randn(len(self.labels), len(self.labels)))
        self.transition.data[label_vocab['E'], :] = -10000  # never transit from the end of a sentence.
        self.transition.data[:, label_vocab['B']] = -10000  # never transit to the beginning of a sentence

    def _viterbi_forward(self, X):
        init_alpha = torch.full((1, len(self.labels)), -10000.)
        init_alpha[0][self.labels['B']] = 0.
        forward_var = init_alpha
        for x in X:
            alpha_t = []
            for next_tag in range(len(self.labels)):
                emit = x[next_tag].view(1, -1).expand(1, len(self.labels))
                trans = self.transition[:, next_tag].view(1, -1)
                next_var = forward_var + trans + emit
                alpha_t.append(log_sum_exp(next_var).view(1))

    def _score(self):
        pass

    def _viterbi_backward(self):
        pass

    def forward(self, X):
        pass


if __name__ == '__main__':
    char_embed = 'lstm'
    train_loader, vocabs, max_sent, max_chars = load_train_data(char_embed=char_embed)
    hidden_size = 128
    encoder = Encoder(vocabs, max_chars, max_sent, hidden_size, char_embed=char_embed)

    for C, S, y in train_loader:
        encoded = encoder(C, S)
        pass
