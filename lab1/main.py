import pandas as pd

from model import ScratchTextClassifier
from preprocess import BOW


def inference(feature_extractor, net, phrase):
    feature = feature_extractor.generate_feature(phrase)
    y_hat = net(feature)
    return y_hat.argmax()


if __name__ == '__main__':
    bow_extractor = BOW()
    net = ScratchTextClassifier([len(bow_extractor.vocab), bow_extractor.num_cls])
    net.load_state('./saved_model/bow/64_1.0_1.pkl')
    phrases = list(pd.read_csv('./data/test.tsv', sep='\t')['Phrase'])
    for phrase in phrases:
        pass

