def extract_data(path):
    data = []
    with open(path, 'r') as f:
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
            idx = i
            if now != 'O':
                length = 1
                while i+1 < len(values) and values[i+1] == now:
                    length += 1
                    i += 1
                if length == 1:
                    values[idx] = 'S-' + now.split('-')[1]
                else:
                    values[idx] = 'B-' + now.split('-')[1]
                    if length > 2:
                        values[idx + 1:idx + length - 1] = ['M-' + now.split('-')[1]] * (length - 1)
                    values[idx+length-1] = 'E-' + now.split('-')[1]

    pass


if __name__ == '__main__':
    extract_data('./data/eng.train')
