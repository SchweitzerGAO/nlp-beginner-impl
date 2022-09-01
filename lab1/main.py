import pandas as pd

from model import ScratchTextClassifier
from preprocess import BOW


def inference(feature_extractor, net, phrase):
    feature = feature_extractor.generate_feature(phrase)
    y_hat = net(feature)
    return y_hat.argmax()


if __name__ == '__main__':
    bow_extractor = BOW(data_path='./proceeded_data/bow.pkl')
    net = ScratchTextClassifier([len(bow_extractor.vocab), bow_extractor.num_cls])
    net.load_state('./saved_model/bow/256_2.5_10.pkl')
    df = pd.read_csv('./data/test.tsv', sep='\t')
    phrases = list(df['Phrase'])
    phrase_id = list(df['PhraseId'])
    predictions = []
    for phrase in phrases:
        predictions.append(inference(bow_extractor, net, phrase))
    with open('prediction.csv', 'w') as f:
        f.write('PhraseId,Sentiment\n')
        for i, y in zip(phrase_id, predictions):
            f.write('{},{}\n'.format(i, y))
