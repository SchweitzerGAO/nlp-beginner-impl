import numpy as np
from preprocess import BOW, NGram, train_test_split, dataloader
from model import ScratchTextClassifier

train_loss = []
train_acc = []
test_acc = []
epochs = []


def accuracy(y, y_hat):
    return None


def test_accuracy(net, test_set):
    return None


def train_epoch(feature_extractor, net, train_set, batch_size, lr):
    gross_loss = 0
    gross_train_acc = 0
    num_batch = 0
    for X, y in dataloader(feature_extractor, train_set, batch_size):
        num_batch += 1
        y_hat = net(X)
        loss, dw, db = net.backward(y)
        net.update_params(lr, dw, db)
        gross_loss += loss
        gross_train_acc += (accuracy(y, y_hat))
    train_loss.append(gross_loss / num_batch)
    train_acc.append(gross_train_acc / num_batch)


def train(feature_extractor, net, train_set, test_set, batch_size, lr, num_epochs):
    for epoch in range(num_epochs):
        epochs.append(epoch + 1)
        train_epoch(feature_extractor, net, train_set, batch_size, lr)
        test_acc.append(test_accuracy(net, test_set))
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
    pass
