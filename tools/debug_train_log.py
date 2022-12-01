import os

import matplotlib.pyplot as plt


def read_log(txt_path):
    now_epoch, best_epoch, best_acc, best_confusion_mat = 0, 0, 0, None
    train_acc_list = []
    val_acc_list = []
    epochs = []
    with open(txt_path, 'r') as rf:
        args = rf.readline()
        for line in rf.readlines():
            line_split = line.split('  ')
            loss = float(line_split[1].split(': ')[1])
            acc = float(line_split[2].split(': ')[1])
            confusion_mat = line_split[3].split(': ')[1]

            if '*Validation' in line:
                now_epoch = int(line_split[0].split(' ')[1])
                time = int(line_split[4].split(': ')[1])
                val_acc_list.append(acc)
                epochs.append(now_epoch)
                if acc > best_acc:
                    best_epoch = now_epoch
                    best_acc = acc
                    best_confusion_mat = confusion_mat
            elif line[0] == '[' and '][' in line:
                now_epoch = int(line_split[0][1:].split('/')[0])
                if len(train_acc_list) == len(val_acc_list):
                    train_acc_list.append(acc)

        print('[{}] best epoch: {}/{}  best acc: {}  best confusion mat: {}'.format(txt_path, best_epoch, now_epoch, best_acc, best_confusion_mat))
        print('Arguments: {}'.format(args.replace('Namespace(', '')[:-2]))
        print()

    return best_epoch, now_epoch, best_acc, train_acc_list, val_acc_list, epochs


if __name__ == '__main__':
    base_dir = '../results/'
    checkpoint_name = ''
    txt_path = os.path.join(base_dir, checkpoint_name, 'log.txt')

    if not os.path.isfile(txt_path):
        raise FileNotFoundError

    best_epoch, now_epoch, best_acc, train_acc_list, val_acc_list, epochs = read_log(txt_path)
    print(len(train_acc_list), len(val_acc_list))

    plt.plot(epochs[1:], train_acc_list[1:])
    plt.plot(epochs[1:], val_acc_list[1:])
    plt.grid(True, 'major', color='k')
    plt.minorticks_on()
    plt.grid(True, 'minor', 'y')
    plt.show()
