import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from model import SeqTagger
from preprocess import load_train_data
from public.misc import grad_clipping

'''
configs
'''
hidden_size = 100
char_embed = 'lstm'
lr = 0.01
batch_size = 128
ep = 10
'''
plot
'''
P = []
R = []
F1 = []

# gpu accessibility
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# preparation
train_loader, vocabs, max_sent, max_chars = load_train_data(char_embed=char_embed)
net = SeqTagger(vocabs, max_chars, max_sent, hidden_size, char_embed)
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)


# plot the P-R-F1 curve
def plot(figure_num, file_name, to_plot, title):
    plt.figure(figure_num)
    plt.plot(to_plot)
    plt.title(title)
    plt.savefig(file_name)


# macro-average of P R F1 score
def macro_ave_P_R_F1(y, y_hat):
    P_sum = 0.
    R_sum = 0.
    F1_sum = 0.
    for tag, tag_hat in zip(y, y_hat):
        TP_TN_FN = dict()
        for tag_idx in tag:
            if TP_TN_FN.get(tag_idx.item(), None) is None:
                TP_TN_FN[tag_idx.item()] = [0, 0, 0]  # TP, FP, FN
        for i, tag_hat_idx in enumerate(tag_hat):
            if tag_hat_idx == tag[i]:
                TP_TN_FN[tag[i].item()][0] += 1
            else:
                if tag_hat_idx in tag:
                    TP_TN_FN[tag[i].item()][2] += 1
                    TP_TN_FN[tag_hat_idx.item()][1] += 1
                else:
                    TP_TN_FN[tag[i].item()][2] += 1
        temp_P_sum = 0.
        temp_R_sum = 0.
        for k in TP_TN_FN.keys():
            temp_P_sum += (TP_TN_FN[k][0] / (TP_TN_FN[k][0] + TP_TN_FN[k][1])
                           if TP_TN_FN[k][0] + TP_TN_FN[k][1] != 0 else 0)
            temp_R_sum += (TP_TN_FN[k][0] / (TP_TN_FN[k][0] + TP_TN_FN[k][2])
                           if TP_TN_FN[k][0] + TP_TN_FN[k][2] != 0 else 0)
        P_sum += (temp_P_sum / len(TP_TN_FN))
        R_sum += (temp_R_sum / len(TP_TN_FN))
        F1_sum += ((2. * P_sum * R_sum) / (P_sum + R_sum) if P_sum + R_sum != 0 else 0)

    return P_sum / batch_size, R_sum / batch_size, F1_sum / batch_size


def train_epoch():
    P_epoch = []
    R_epoch = []
    F1_epoch = []

    for C, S, y in train_loader:
        C = C.to(device)
        S = S.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        batch_enc_output, y_hat = net(C, S)
        loss = net.decoder.neg_log_likelihood(batch_enc_output, y)
        loss.backward()
        grad_clipping(net, 5)
        optimizer.step()
        P_batch, R_batch, F1_batch = macro_ave_P_R_F1(y, y_hat)
        P_epoch.append(P_batch)
        R_epoch.append(R_batch)
        F1_epoch.append(F1_batch)
    return np.mean(np.array(P_epoch)), np.mean(np.array(R_epoch)), np.mean(np.array(F1_epoch))


def train(save_path):
    net.train()
    for i in range(ep):
        p, r, f1 = train_epoch()
        print(
            f'Epoch({i + 1}/{ep}): '
            f'precision: {np.round(p, 4)}; '
            f'recall: {np.round(r, 4)}; '
            f'f1 score: {np.round(f1, 4)}')
        P.append(p)
        R.append(r)
        F1.append(f1)
    torch.save(net.state_dict(), save_path + f'/{char_embed}_{ep}_{batch_size}.pt)')
    plot(1, f'/{char_embed}_{ep}_{batch_size}_P.png', P, f'Precision_{char_embed}_{ep}_{batch_size}')
    plot(2, f'/{char_embed}_{ep}_{batch_size}_R.png', R, f'Recall_{char_embed}_{ep}_{batch_size}')
    plot(3, f'/{char_embed}_{ep}_{batch_size}_F1.png', F1, f'F1_{char_embed}_{ep}_{batch_size}')


if __name__ == '__main__':
    train('./saved_models')
