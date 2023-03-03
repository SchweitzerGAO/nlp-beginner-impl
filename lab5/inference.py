import torch
from model import TextGenerator
from preprocess import load_corpus

# GPU accessibility
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

_, vocab = load_corpus()

embedding_size = 100
hidden_size = 256
pred_length = 14
net = TextGenerator(embedding_size, hidden_size, len(vocab))


def inference(start, weight_path):
    net.eval()
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    state = net.begin_state(device=device)
    outputs = [vocab[start[0]]]

    def get_input():
        return torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    if len(start) > 1:  # switch the state to the last character of input
        for y in start[1:]:
            _, state = net(get_input(), state)
            outputs.append(vocab[y])
    for _ in range(pred_length - len(start)):
        y, state = net(get_input(), state)
        y = torch.argmax(y, dim=1)
        outputs.append(int(y[0, 0]))
    outputs = [vocab.to_tokens(i) for i in outputs]
    return outputs


if __name__ == '__main__':
    sent_len = pred_length // 2
    outputs = inference('帆', 'saved_model/gru/100_64_10.pt')
    for i in range(0, len(outputs), sent_len):
        print(''.join(outputs[i:i+sent_len]))
