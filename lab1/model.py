import numpy as np
from preprocess import BOW, train_test_split, dataloader
import pickle as pkl

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
    return np.mean(np.sum(-y * np.log(y_hat+1e-7), axis=-1)), y_hat - y  # the second one is the derivative of softmax


# relu
def relu(x):
    return np.maximum(0, x)


# derivative of relu
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
    def __init__(self, net_arch):
        assert net_arch is not None
        self.net_arch = net_arch
        self.weights = [np.random.randn(in_channel, out_channel)
                        for in_channel, out_channel
                        in zip(self.net_arch[:-1], self.net_arch[1:])]
        self.biases = [np.zeros((1, channel)) for channel in self.net_arch[1:]]
        self.P = []
        self.X = []

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.X.clear()
        self.P.clear()
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
        for layer_idx in range(len(self.weights) - 1, -1, -1):
            x = self.X[layer_idx]
            x = x.swapaxes(1, 2)
            db[layer_idx] = np.sum(delta, axis=0) / batch_size
            dw[layer_idx] = np.sum(np.einsum('ijk,ikn->ijn', x, delta), axis=0) / batch_size
            if layer_idx >= 1:
                delta = (delta @ self.weights[layer_idx].T) * relu_prime(self.P[layer_idx - 1])
        return loss, dw, db

    def update_params(self, lr, dw, db, l1=None, l2=None):
        if l1 is not None:
            self.weights = [W - lr * (grad_w + l1 * np.sign(W)) for W, grad_w in zip(self.weights, dw)]
            self.biases = [b - lr * grad_b for b, grad_b in zip(self.biases, db)]
        elif l2 is not None:
            self.weights = [W - lr * (grad_w + l2 * W) for W, grad_w in zip(self.weights, dw)]
            self.biases = [b - lr * grad_b for b, grad_b in zip(self.biases, db)]
        else:
            self.weights = [W - lr * grad_w for W, grad_w in zip(self.weights, dw)]
            self.biases = [b - lr * grad_b for b, grad_b in zip(self.biases, db)]

    def load_state(self, path):
        with open(path, 'rb') as rf:
            params = pkl.load(rf)
        self.weights = params['weights']
        self.biases = params['biases']


'''
test code
'''
if __name__ == '__main__':
    batch_size = 32
    lr = 0.1
    bow = BOW()
    train_set, test_set, train_label, test_label = train_test_split(bow)
    net = ScratchTextClassifier([len(bow.vocab), bow.num_cls])
    for X, y in dataloader(bow, test_set, test_label, batch_size):
        X = X.reshape((batch_size, 1, -1))
        y = y.reshape((batch_size, 1, -1))
        y_hat = net(X)
        loss, dw, db = net.backward(y)
        print(loss.shape)
        net.update_params(lr, dw, db)
        break
