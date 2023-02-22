import torch
from torch import nn

from preprocess import load_poems


class TextGenerator(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, mode='gru', num_layers=1):
        super().__init__()
        self.num_hiddens = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        if mode == 'gru':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif mode == 'lstm':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise NotImplementedError
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, sentence, state):
        embedded = self.embed(sentence)
        out, state = self.rnn(embedded, state)
        out = self.dense(out)
        return out.permute(0, 2, 1), state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            return (torch.zeros((self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))


if __name__ == '__main__':
    batch_size = 64
    num_steps = 10
    embedding_size = 100
    hidden_size = 256
    data, voc = load_poems(batch_size, num_steps)
    net = TextGenerator(embedding_size, hidden_size, len(voc))
    state = net.begin_state(batch_size=batch_size, device='cpu')
    for X, y in data:
        y_hat, state = net(X, state)
        test = torch.argmax(y_hat, dim=1)
        pass
