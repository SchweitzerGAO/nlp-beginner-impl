import re


def read_data():
    with open('data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [re.sub('[^\u4e00-\u9fa5]+', '', line).strip().lower() for line in lines]


def tokenize(lines):
    return [list(line) for line in lines]


if __name__ == '__main__':
    lines = read_data()
    tokens = tokenize(lines)
    for i in range(10):
        print(tokens[i])
