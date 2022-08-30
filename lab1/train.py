import pickle as pkl

from matplotlib import pyplot as plt

from model import ScratchTextClassifier
from preprocess import BOW, NGram, train_test_split, dataloader

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
    for X, y in dataloader(feature_extractor, test_set, test_label, batch_size):
        num_batch += 1
        X = X.reshape((batch_size, 1, -1))
        y = y.reshape((batch_size, 1, -1))
        y_hat = net(X)
        gross_acc += accuracy(y, y_hat)
    return gross_acc / num_batch


def train_epoch(feature_extractor, net, train_set, train_label, batch_size, lr, l1=None, l2=None):
    gross_loss = 0.
    gross_train_acc = 0.
    num_batch = 0
    for X, y in dataloader(feature_extractor, train_set, train_label, batch_size):
        num_batch += 1
        X = X.reshape((batch_size, 1, -1))
        y = y.reshape((batch_size, 1, -1))
        y_hat = net(X)
        loss, dw, db = net.backward(y)
        net.update_params(lr, dw, db, l1=l1, l2=l2)
        gross_loss += loss
        gross_train_acc += (accuracy(y, y_hat))
    train_loss.append(gross_loss / num_batch)
    train_acc.append(gross_train_acc / num_batch)


def train(feature_extractor, net, train_set, train_label, test_set, test_label, batch_size, lr, num_epochs, l1=None,
          l2=None, gamma=None):
    for epoch in range(num_epochs):
        epochs.append(epoch + 1)
        train_epoch(feature_extractor, net, train_set, train_label, batch_size, lr, l1, l2)
        lr = round(lr * gamma, 2) if gamma is not None else lr
        test_acc.append(test_accuracy(feature_extractor, net, test_set, test_label, batch_size))
        print(
            f'Epoch({epoch + 1}/{num_epochs}): '
            f'loss:{round(train_loss[epoch], 4)}; '
            f'train_acc:{round(train_acc[epoch] * 100., 4)} %; '
            f'test_acc:{round(test_acc[epoch] * 100., 4)} %')
        if (epoch + 1) % 10 == 0:
            plot_train(epoch + 1)
            params = dict()
            params['weights'] = net.weights
            params['biases'] = net.biases
            file_path = './saved_model'
            if isinstance(feature_extractor, BOW):
                file_path += f'/bow/{batch_size}_{lr}_{epoch + 1}.pkl'
            elif isinstance(feature_extractor, NGram):
                file_path += f'/ngram/{batch_size}_{lr}_{epoch + 1}.pkl'
            with open(file_path, 'wb') as wf:
                pkl.dump(params, wf)


def plot_train(epoch):
    # train loss figure
    plt.subplot(3, 1, 1)
    plt.title('Loss')
    plt.plot(epochs, train_loss, c='r', label='loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    # train acc
    plt.subplot(3, 1, 2)
    plt.title('Train Accuracy')
    plt.plot(epochs, train_acc, c='r', label=' train accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('train accuracy')
    plt.legend()

    # test acc
    plt.subplot(3, 1, 3)
    plt.title('Test Accuracy')
    plt.plot(epochs, test_acc, c='r', label='test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('test accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./train_{epoch}.png')


if __name__ == '__main__':
    lr = 5.
    num_epochs = 50
    batch_size = 128
    # bow_extractor = BOW(max_features=5000, data_path='./proceeded_data/bow_5000.pkl')
    bigram_5000 = NGram(max_features=5000, n=2, data_path='./proceeded_data/bigram_5000.pkl')
    net = ScratchTextClassifier([len(bigram_5000.vocab), 256, bigram_5000.num_cls])
    train_set, test_set, train_label, test_label = train_test_split(bigram_5000)
    train(bigram_5000, net, train_set, train_label, test_set, test_label, batch_size, lr, num_epochs, gamma=0.9)
