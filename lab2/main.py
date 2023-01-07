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
vec_dim = 5
out_channels = 5


def inference(net, vec_path, weight_path):
    predictions = []
    with open(vec_path, 'rb') as f:
        dic = pkl.load(f)
    net.load_state_dict(torch.load(weight_path))
    if has_cuda:
        net = net.cuda()
    with torch.no_grad():
        for sent in sents:
            wv = sentence_to_vector(sent, dic, vec_dim)
            if has_cuda:
                net = net.cuda()
            out = net(wv)
            pred = torch.max(out, dim=1)[1]
            predictions.append(np.array(pred.cpu())[0])
            print(predictions[-1])
    with open('prediction.csv', 'w') as f:
        f.write('PhraseId,Sentiment\n')
        for i, y in zip(phrase_id, predictions):
            f.write('{},{}\n'.format(i, y))


if __name__ == '__main__':
    model = TextCNN(vec_dim, out_channels)
    inference(model, './word_vectors/random.pkl', './saved_models/textcnn_random/10_128.pt')
