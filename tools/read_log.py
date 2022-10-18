import os


def read_log(txt_path):
    now_epoch, best_epoch, best_acc, best_confusion_mat = 0, 0, 0, None
    with open(txt_path, 'r') as rf:
        for line in rf.readlines():
            if '*Validation' in line:
                line_split = line.split('  ')
                loss = float(line_split[1].split(': ')[1])
                acc = float(line_split[2].split(': ')[1])
                confusion_mat = line_split[3].split(': ')[1]
                time = int(line_split[4].split(': ')[1])
                if acc > best_acc:
                    best_epoch = now_epoch
                    best_acc = acc
                    best_confusion_mat = confusion_mat
            elif line[0] == '[':
                now_epoch = int(line[1:].split('/')[0])

    print('[{}] best epoch: {}/{}  best acc: {}  best confusion mat: {}'.format(txt_path, best_epoch, now_epoch, best_acc, best_confusion_mat))
    return best_epoch, now_epoch, best_acc


if __name__ == '__main__':
    base_dir = '../results'
    for test_name in sorted(os.listdir(base_dir)):
        txt_path = os.path.join(base_dir, test_name, 'log.txt')
        if not os.path.isfile(txt_path):
            continue
        read_log(txt_path)
