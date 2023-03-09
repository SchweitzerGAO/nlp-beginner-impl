

"""
extract useful data from the original dataset
"""
def extract_data(pathr, pathw):
    data = []
    with open(pathr, 'r') as f:
        keys = []
        values = []
        for i, line in enumerate(f):
            if i >= 2:
                if line != '\n':
                    kk = line.split()[0].lower()
                    keys.append(kk)
                    vv = line.split()[3]
                    values.append(vv)
                else:
                    data.append([keys, values])
                    keys = []
                    values = []
    for sent_dict in data:
        values = sent_dict[1]
        for i in range(len(values)):
            now = values[i]
            if now[0] == 'I':
                length = 1
                idx = i
                while i + 1 < len(values) and values[i + 1] == now:
                    length += 1
                    i += 1
                if length == 1:
                    values[idx] = 'S-' + now.split('-')[1]
                else:
                    values[idx] = 'B-' + now.split('-')[1]
                    for j in range(idx + 1, idx + length - 1):
                        values[j] = 'M-' + now.split('-')[1]
                    values[idx + length - 1] = 'E-' + now.split('-')[1]

    with open(pathw, 'w') as f:
        for sent in data:
            for i in range(len(sent[0])):
                f.write(sent[0][i] + ' ' + sent[1][i] + '\n')
            f.write('\n')


if __name__ == '__main__':
    extract_data('./data/eng.testb', './data/testb.txt')
