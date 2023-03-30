import torch
import torch.nn as nn

from preprocess import load_train_data
from public.misc import PretrainedEmbedding


def argmax(data):
    _, max_idx = torch.max(data, dim=1)
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
        self.dropout = nn.Dropout(0.5)

    def forward(self, C, S):
        embed = self.embed(C, S)
        out, _ = self.bi_lstm_enc(embed)
        out = self.hidden2tag(self.dropout(out))
        return out


class CRFDecoder(nn.Module):
    def __init__(self, label_vocab):
        super().__init__()
        self.labels = label_vocab
        # transition[i,j] is the score of transition from j to i (different to the implementation in
        # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py)
        self.transition = nn.Parameter(torch.randn(len(self.labels), len(self.labels)))
        self.transition.data[label_vocab['<eos>'], :] = -10000  # never transit from the end of a sentence.
        self.transition.data[:, label_vocab['<bos>']] = -10000  # never transit to the beginning of a sentence

    def _forward_alg(self, batch_enc_output):
        result = []
        for sent in batch_enc_output:
            init_alpha = torch.full((1, len(self.labels)), -10000.)
            init_alpha[0, self.labels['<bos>']] = 0.
            forward_var = init_alpha
            for word in sent:
                alpha_t = []
                for next_tag in range(len(self.labels)):
                    emit = word[next_tag].view(1, -1).expand(1, len(self.labels))
                    trans = self.transition[:, next_tag].view(1, -1)
                    next_var = forward_var + trans + emit
                    alpha_t.append(log_sum_exp(next_var).view(1))
                    # alpha_t.append(next_var.view(1))
                forward_var = torch.cat(alpha_t).view(1, -1)
            terminal_var = forward_var + self.transition[:, self.labels['<eos>']]
            result.append(log_sum_exp(terminal_var).view(1))
            # result.append(terminal_var.view(1))
        return torch.cat(result).view(1, -1)

    def _score(self, batch_enc_output, y):
        scores = []
        for i, sent in enumerate(batch_enc_output):
            score = torch.zeros(1)
            for j, word in enumerate(sent[0:-1]):
                score += self.transition[y[i, j], y[i, j + 1]] + word[y[i, j + 1]]
            score += self.transition[y[i, -1], self.labels['<eos>']]
            scores.append(score)
        return torch.cat(scores).view(1, -1)

    def _viterbi_decode(self, batch_enc_output):
        best_paths = []
        path_scores = []
        for sent in batch_enc_output:
            back_tracer = []
            init_viterbi = torch.full((1, len(self.labels)), -10000.)
            init_viterbi[0, self.labels['<bos>']] = 0.
            forward_var = init_viterbi
            for word in sent:
                best_ptr_t = []
                viterbi_t = []
                for next_tag in range(len(self.labels)):
                    next_t = forward_var + self.transition[:, next_tag].view(1, -1)
                    best_tag = argmax(next_t)
                    best_ptr_t.append(best_tag)
                    viterbi_t.append(next_t[0, best_tag].view(1))
                forward_var = (torch.cat(viterbi_t) + word).view(1, -1)
                back_tracer.append(best_ptr_t)
            terminal_var = forward_var + self.transition[:, self.labels['<eos>']].view(1, -1)
            best_tag = argmax(terminal_var)
            path_scores.append(terminal_var[0, best_tag].view(1))

            best_path = [best_tag]
            for best_ptr_t in reversed(back_tracer):
                best_tag = best_ptr_t[best_tag]
                best_path.append(best_tag)
            start = best_path.pop()
            assert start == self.labels['<bos>']
            best_path.reverse()
            best_paths.append(torch.tensor(best_path))
        return torch.cat(path_scores), torch.cat(best_paths).view(-1, best_paths[0].size(0))

    def neg_log_likelihood(self, batch_enc_output, y):
        vf = self._forward_alg(batch_enc_output)
        score = self._score(batch_enc_output, y)
        return torch.mean(vf - score)

    def forward(self, X):
        scores, best_paths = self._viterbi_decode(X)
        return scores, best_paths


class SeqTagger(nn.Module):
    def __init__(self, vocabs, max_chars, max_sent, hidden_size, char_embed):
        super().__init__()
        self.encoder = Encoder(vocabs, max_chars, max_sent, hidden_size, char_embed=char_embed)
        self.decoder = CRFDecoder(vocabs[2])

    def forward(self, C, S):
        encoded = self.encoder(C,S)
        _, best_paths = self.decoder(encoded)
        return encoded, best_paths


if __name__ == '__main__':
    char_embed = 'lstm'
    train_loader, vocabs, max_sent, max_chars = load_train_data(char_embed=char_embed)
    hidden_size = 128
    encoder = Encoder(vocabs, max_chars, max_sent, hidden_size, char_embed=char_embed)
    decoder = CRFDecoder(vocabs[2])

    for C, S, y in train_loader:
        encoded = encoder(C, S)
        # vf = decoder.forward_alg(encoded)
        # scores = decoder.score(encoded, y)
        _, best_paths = decoder(encoded)
        pass
