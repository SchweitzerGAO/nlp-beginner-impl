import numpy as np
from preprocess import BOW, NGram, train_test_split, dataloader
from model import ScratchTextClassifier

train_loss = []
train_acc = []
test_acc = []
epochs = []


def accuracy(y, y_hat):  # y is gt and y_hat is prediction
    y_hat = y_hat.argmax(axis=2)
    y = y.argmax(axis=2)
    return (y_hat == y).sum() / len(y)


def test_accuracy(feature_extractor, net, test_set, test_label, batch_size):
    gross_acc = 0.
    num_batch = 0
    for X, y in dataloader(feature_extractor, test_set, test_label, batch_size, shuffle=False):
        num_batch += 1
        y_hat = net(X)
        gross_acc += accuracy(y, y_hat)
    return gross_acc / num_batch


def train_epoch(feature_extractor, net, train_set, train_label, batch_size, lr):
    gross_loss = 0.
    gross_train_acc = 0.
    num_batch = 0
    for X, y in dataloader(feature_extractor, train_set, train_label, batch_size):
        num_batch += 1
        X = X.reshape((batch_size, 1, -1))
        y = y.reshape((batch_size, 1, -1))
        y_hat = net(X)
        loss, dw, db = net.backward(y)
        net.update_params(lr, dw, db)
        gross_loss += loss
        gross_train_acc += (accuracy(y, y_hat))
    train_loss.append(gross_loss / num_batch)
    train_acc.append(gross_train_acc / num_batch)


def train(feature_extractor, net, train_set, train_label, test_set, test_label, batch_size, lr, num_epochs):
    for epoch in range(num_epochs):
        epochs.append(epoch + 1)
        train_epoch(feature_extractor, net, train_set, train_label, batch_size, lr)
        test_acc.append(test_accuracy(net, test_set, test_label))
        print(
            f'Epoch({epoch + 1}/{num_epochs}): loss:{train_loss[epoch]}; train_acc:{train_acc[epoch] * 100.}%; '
            f'test_acc:{test_acc[epoch] * 100.}%')
        if (epoch + 1) % 100 == 0:
            params = dict()
            params['weights'] = net.weights
            params['biases'] = net.biases
            file_path = './saved_model'
            if isinstance(feature_extractor, BOW):
                file_path += f'/bow/{batch_size}_{lr}_{epoch + 1}.npy'
            else:
                file_path += f'/ngram/{batch_size}_{lr}_{epoch + 1}.npy'
            with open(file_path, 'wb') as wf:
                np.save(wf, params)


if __name__ == '__main__':
    lr = 0.1
    num_epochs = 10
    batch_size = 128
    bow_extractor = BOW()
    net = ScratchTextClassifier(len(bow_extractor.vocab), bow_extractor.num_cls)
    train_set, test_set, train_label, test_label = train_test_split(bow_extractor)
    train(bow_extractor, net, train_set, train_label, test_set, test_label, batch_size, lr, num_epochs)
