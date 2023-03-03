import torch
import torch.nn.functional as F
from model import ESIM
import pickle as pkl

from preprocess import tokenize_word
from public.misc import truncate_pad

embed_size = 100
length = 50
hidden_size_lstm = 128
hidden_size_dense = 128


def prediction(weight_path, premise, hypothesis):
    predictions = ['entailment', 'contradiction', 'neutral']
    with open('./train_vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    tokens_p = tokenize_word(premise.lower())
    tokens_h = tokenize_word(hypothesis.lower())
    p = truncate_pad(vocab[tokens_p], length, vocab['<pad>']).unsqueeze(0)
    h = truncate_pad(vocab[tokens_h], length, vocab['<pad>']).unsqueeze(0)
    net = ESIM(vocab, embed_size, length, hidden_size_lstm, hidden_size_dense, output_size=3)
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()
    y_hat = net(p, h)
    pred = torch.max(F.softmax(y_hat, dim=1), dim=1)[1][0]
    print(predictions[pred])


if __name__ == '__main__':
    premise = 'My mother does not like my girlfriend'
    hypothesis = 'My mother likes my girlfriend'
    prediction('./saved_models/20_256.pt', premise, hypothesis)
