# nlp-beginner-impl

[TOC]

---

## Lab 1

*test accuracy here actually means validation accuracy*

An MLP for text classification implementation from scratch only by NumPy is required in Lab 1

training plot:

![](./lab1/train50.png)

The accuracy on the corresponding [Kaggle comepetition](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews) is bad, just 19%

--- 

## Lab 2

A Pytorch version of Lab 1 with CNN and RNN classifier is required in Lab 2, whose initialization shall be random or by pre-trained GLoVe vectors. Here are some results

*test accuracy here actually means validation accuracy*

*epoch for all experiments is 50*

**CNN train plot**

- TextCNN([paper](https://arxiv.org/abs/1408.5882)) with GLoVe 50d

*learning rate 0.001*

*dropout 0.5*

*batch size 128*

*weight decay 0.001*

![](./lab2/plots/cnn_50.png)

The best performance on the same competition in Lab 1 reached 58.3%

**RNN training plots**

1. GRU with GLoVe 50d

*learning rate 0.001*

*batch size 64*

*weight decay 1e-5*

<img title="" src="./lab2/plots/rnn_gru_50.png" alt="" data-align="inline">

The best performance on the same competition in Lab 1 reached 60.3%

2. LSTM with GLoVe 100d

*learning rate 0.001*

*batch size 64*

*weight decay 1e-5*

![](./lab2/plots/rnn_lstm_100.png)

but the best performance is 59.2%, a little bit worse than GRU

Further experiments are needed to find out whether this is because of the change of NN structur or the dimension of word vector

3. GRU with GLoVe 100d

*learning rate 0.001*

*batch size 64*

*weight decay 1e-5*

![](./lab2/plots/rnn_gru_100.png)

The best performance on the same competition in Lab 1 reached 61.3%

*By comparing experiment 2 and 3, it can be concluded that it is the structure of the network that influences the result.*

*A possible reason for this is the parameters in the hidden layer of LSTM(a tuple of 2 tensors) is more than that in GRU(1 tensor). So it is prone to overfit.*

- Bi-GRU with GLoVe 100d

- Bi-LSTM with GLoVe 100d

## Lab 3

---

## Lab 4

---

## Lab 5
