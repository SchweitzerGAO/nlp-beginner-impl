import numpy as np
from preprocess import BOW, train_test_split, dataloader

'''
utils
'''


# softmax
def softmax(x):
    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp, axis=-1, keepdims=True)
    return x_exp / x_exp_sum


# cross entropy
def cross_entropy(y, y_hat):  # y is gt and y_hat is prediction
    return np.sum(-y * np.log(y_hat), axis=-1), y_hat - y  # the second one is the derivative of softmax


# relu
def relu(x):
    return np.maximum(0, x)


def relu_prime(p):
    prime = np.zeros_like(p)
    prime[p > 0] = 1
    prime[p < 0] = 0
    prime[p == 0] = 0.5
    return prime


'''
an MLP classifier
'''


class ScratchTextClassifier:
    def __init__(self, len_vocab, num_cls, num_hidden=256):
        self.len_vocab = len_vocab
        self.num_cls = num_cls
        self.num_hidden = num_hidden
        self.weights = [np.random.randn(self.len_vocab, self.num_hidden),
                        np.random.randn(self.num_hidden, self.num_cls)]
        self.biases = [np.zeros((1, self.num_hidden)), np.zeros((1, self.num_cls))]
        self.P = []
        self.X = []

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.X.clear()
        self.X.append(x)
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            p = x @ W + b
            self.P.append(p)
            if layer_idx < len(self.weights) - 1:
                x = relu(p)
            else:
                x = softmax(p)
            self.X.append(x)
        return self.X[-1]

    def backward(self, y):
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        loss, delta = cross_entropy(y, self.X[-1])
        batch_size = len(y)
        for layer_idx in range(len(self.X) - 2, -1, -1):
            x = self.X[layer_idx]
            x = x.swapaxes(1, 2)
            db[layer_idx] = np.sum(delta, axis=0) / batch_size
            dw[layer_idx] = np.sum(np.einsum('ijk,ikn->ijn', x, delta), axis=0) / batch_size
            if layer_idx >= 1:
                delta = (delta @ self.weights[layer_idx].T) * relu_prime(self.P[layer_idx - 1])
        return loss, dw, db

    def update_params(self, lr, dw, db):
        self.weights = [W - lr * grad_w for W, grad_w in zip(self.weights, dw)]
        self.biases = [b - lr * grad_b for b, grad_b in zip(self.biases, db)]


'''
test code
'''
if __name__ == '__main__':
    batch_size = 32
    lr = 0.1
    bow = BOW()
    train_set, test_set, train_label, test_label = train_test_split(bow)
    net = ScratchTextClassifier(len(bow.vocab), bow.num_cls)
    for X, y in dataloader(bow, test_set, test_label,batch_size):
        X = X.reshape((batch_size, 1, -1))
        y = y.reshape((batch_size, 1, -1))
        y_hat = net(X)
        loss, dw, db = net.backward(y)
        net.update_params(lr, dw, db)
        break
