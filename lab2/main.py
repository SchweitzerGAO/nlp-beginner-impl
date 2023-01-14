import numpy as np
import torch
from model import TextCNN
import pandas as pd
import pickle as pkl
from preprocess import sentence_to_vector

df = pd.read_csv('../lab1/data/test.tsv', sep='\t')
sents = list(df['Phrase'])
phrase_id = list(df['PhraseId'])
has_cuda = torch.cuda.is_available()
vec_dim = 50
out_channels = 5


def inference_cnn(net, vec_path, weight_path):
    predictions = []
    with open(vec_path, 'rb') as f:
        dic = pkl.load(f)
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()
    if has_cuda:
        net = net.cuda()
    with torch.no_grad():
        for sent in sents:
            wv = sentence_to_vector(sent, dic, vec_dim)
            if has_cuda:
                net = net.cuda()
                wv = wv.cuda()
            wv = wv.unsqueeze(0)
            out = net(wv)
            pred = torch.max(out, dim=1)[1]
            predictions.append(np.array(pred.cpu())[0])
    with open('prediction.csv', 'w') as f:
        f.write('PhraseId,Sentiment\n')
        for i, y in zip(phrase_id, predictions):
            f.write('{},{}\n'.format(i, y))


def inference_rnn(net, vec_path, weight_path):
    pass

if __name__ == '__main__':
    model = TextCNN(vec_dim, out_channels)
    inference_cnn(model, './word_vectors/glove_6B_50d.pkl', './saved_models/textcnn_random/10_128.pt')
