import numpy as np


# softmax
def softmax(x):
    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp, axis=-1, keepdims=True)
    return x_exp / x_exp_sum


# cross entropy
def cross_entropy(y, y_hat):  # y is gt and y_hat is prediction
    return -np.sum(y * np.log(y_hat), axis=-1)


# relu
def relu(x):
    return np.max(0, x)


class ScratchTextClassifier:
    def __init__(self, len_vocab, num_cls, num_hidden=256):
        self.len_vocab = len_vocab
        self.num_cls = num_cls
        self.num_hidden = num_hidden
        self.weights = [np.random.randn(self.len_vocab, self.num_hidden),
                        np.random.randn(self.num_hidden, self.num_cls)]
        self.biases = [np.zeros(1, self.num_hidden), np.zeros(1, self.num_cls)]
        self.output = []
        self.activated_output = []

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for W, b in zip(self.weights, self.biases):
            X = X @ W + b
        X = softmax(X)
        self.output.append(X)
        activated = relu(X)
        self.activated_output.append(activated)
        return activated

    def backward(self, y):
        dw = []
        db = []
        return dw, db

    def update_params(self, lr, dw, db):
        self.weights = [W - lr * grad_w for W, grad_w in zip(self.weights, dw)]
        self.biases = [b - lr * grad_b for b, grad_b in zip(self.biases, db)]


'''
test code
'''
if __name__ == '__main__':
    pass
