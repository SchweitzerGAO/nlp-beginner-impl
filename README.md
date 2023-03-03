# nlp-beginner-impl

This is a possible implementation of https://github.com/FudanNLP/nlp-beginner

[TOC]

---

## Lab 1

*test accuracy here actually means validation accuracy*

An MLP for text classification implementation from scratch only by NumPy is required in Lab 1

training plot:

![](./lab1/train50.png)

*The accuracy on the corresponding [Kaggle comepetition](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews) is bad, just 19%*

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

4. Bi-GRU with GLoVe 100d

*learning rate 0.001*

*batch size 64*

*weight decay 1e-4*

![](./lab2/plots/rnn_bigru_100.png)

The best performance on the same competition in Lab 1 is 60.7%

*I do not know why Bi-GRU or Bi-LSTM won't improve performance. Maybe there is better hyper-parameter combinations.*

*Besides I also tried gradient clipping but it will do bad to this lab because the amount of training data is not large(at $10^5$ level). Gradient boom won't appear.*

*This is the end of lab 1 & 2*

---

## Lab 3

This is an implementation of the ESIM model of [this paper](https://arxiv.org/abs/1609.06038)

Hyper parameters:

```py
batch_size = 256
embed_size = 100
sentence_length = 50
hidden_size_lstm = 128
hidden_size_dense = 128
lr = 4e-4
epoch = 20
dropout = 0.5
```

I trained the model 30 epochs using SNLI dataset. But when tested, it seems 10 epochs is enough, more epochs could cause overfit. 

The loss curve:

![](./lab3/plots/30_loss.png)

The accuracy & loss curve:

![](./lab3/plots/30_acc.png)

Test accuracies:

| Epoch | Acc     |
|:-----:|:-------:|
| 10    | 85.50 % |
| 20    | 83.69 % |
| 30    | 83.16 % |

Customized test is available in `prediction.py`, You can play with this.

*Concatenating information produced by LSTM is useful. Attention is strong*

---

## Lab 4

To be done

---

## Lab 5

1. 花丛狭路尘
   间黯将暮云
   间月色明如
   素鸳鸯池上
   两两飞凤凰
   楼下双双度
   物色正如此
   佳期那不顾

2. 明月夜玉殿莓苔
   青宫女晚知曙祠
   官朝见星空梁簇
   画戟阴井敲铜瓶
   中使日夜继惟王
   心不宁岂徒恤备
   享尚谓求无形孝
   理敦国政神凝推

*hyper-parameters*

`batch_size = 64`

`num_steps = 10 or 14`

`embedding_size = 100`

`hidden_size = 256`

`lr = 0.001`

There are poems generated by the GRU model. Somehow nonsense.(Copying the dataset)

I tried the time steps of 10(for 5-charactered-sentence poem) and 14(for 7-charactered-sentence poem)

The perplexity of time step 10:

![](./lab5/plots/100_gru_10.png)

The perplexity of time step 14:

![](./lab5/plots/100_gru_14.png)

*I think it is better to use seq2seq. I will try if time allows.*

*This is the end of lab 5*
