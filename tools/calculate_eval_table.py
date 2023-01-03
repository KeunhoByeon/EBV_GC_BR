import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = '../results/20221129163053_color_noise'
# BASE_DIR = './results/20221201192712_base_model'
CALC_TYPES = ('tissue', 'tumor', 'size')


def load_eval_csv_data(csv_path):
    data = {}
    with open(csv_path, 'r') as rf:
        header = rf.readline().replace('\n', '').split(",")

        for line in rf.readlines():
            line_split = line.replace('\n', '').split(",")
            filename = line_split[0]
            x, y = int(line_split[1]), int(line_split[2])
            pred = int(line_split[3])
            target = line_split[4]
            if target == "Positive":
                target = 1
            elif target == "Negative":
                target = 2
            else:
                print(line_split)
                raise AssertionError

            file_index = filename.split("_patch_")[0]

            if file_index not in data.keys():
                data[file_index] = {"target": target, "preds": []}
            elif data[file_index]["target"] != target:
                print(line_split)
                raise AssertionError

            data[file_index]["preds"].append(pred)

    return data


def get_result(data, threshold=0.1, calc_type="tissue", return_arr=False):
    correct_num, total_num = 0, 0

    output_arr = [[] for _ in range(3)]

    confusion_mat = [[0 for _ in range(3)] for _ in range(3)]
    for file_index, data_dict in data.items():
        target = data_dict["target"]
        unq, unq_cnt = np.unique(data_dict["preds"], return_counts=True)
        cnt_dict = dict(zip(unq, unq_cnt))

        cnt_0 = 0 if 0 not in cnt_dict else cnt_dict[0]
        cnt_1 = 0 if 1 not in cnt_dict else cnt_dict[1]
        cnt_2 = 0 if 2 not in cnt_dict else cnt_dict[2]

        if calc_type == "tissue":
            if cnt_1 / (cnt_0 + cnt_1 + cnt_2) > threshold:
                total_pred = 1
            else:
                total_pred = 2
            output_arr[target].append(cnt_1 / (cnt_0 + cnt_1 + cnt_2))
        elif calc_type == 'tumor':
            if cnt_1 / (cnt_1 + cnt_2) > threshold:
                total_pred = 1
            else:
                total_pred = 2
            output_arr[target].append(cnt_1 / (cnt_1 + cnt_2))
        elif calc_type == 'size':
            if cnt_1 > threshold:
                total_pred = 1
            else:
                total_pred = 2
            output_arr[target].append(cnt_1)
        else:
            print("Calc type {} not yet implemented".format(calc_type))
            raise AssertionError

        confusion_mat[target][total_pred] += 1

        total_num += 1
        if target == total_pred:
            correct_num += 1

    if return_arr:
        return output_arr
    return confusion_mat


if __name__ == "__main__":
    result_csv_path = BASE_DIR + '/table.csv'
    result_csv_wf = open(result_csv_path, 'w')
    result_csv_wf.write('csv_path, calc_type, threshold, sensitivity, specificity, ss_2, confusion_mat[1][1], confusion_mat[1][2], confusion_mat[2][1], confusion_mat[2][2]\n')

    csv_paths = []
    for eval_name in os.listdir(BASE_DIR):
        if 'eval_' not in eval_name:
            continue
        for epoch in os.listdir(os.path.join(BASE_DIR, eval_name)):
            csv_path = os.path.join(BASE_DIR, eval_name, epoch, 'results.csv')
            if os.path.isfile(csv_path):
                csv_paths.append(csv_path)
    csv_paths.sort()

    best_ss_2_dict = {}
    threshold_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for csv_path in csv_paths:
        for calc_type in CALC_TYPES:
            data = load_eval_csv_data(csv_path)
            print(csv_path)

            best_ss_2_dict['{}_{}'.format(csv_path, calc_type)] = (0, 0, [])
            for threshold in threshold_percentages:
                if calc_type == 'size':
                    threshold = threshold * 2000
                confusion_mat = get_result(data, threshold=threshold, calc_type=calc_type)
                tp = confusion_mat[1][1]
                fn = confusion_mat[1][2]
                fp = confusion_mat[2][1]
                tn = confusion_mat[2][2]
                accuracy = (tp + tn) / (tp + fn + fp + tn)
                precision = tp / (tp + fp) if (tp + fp) != 0 else np.nan
                recall = tp / (tp + fn)
                sensitivity = tp / (tp + fn)
                specificity = tn / (fp + tn)
                ss_2 = (sensitivity + specificity) / 2
                f1_score = 2 * recall * precision / (recall + precision)
                print("threshold {}: {}".format(threshold, confusion_mat))
                # print("accuracy: {:0.3f}".format(accuracy))
                # print("precision: {:0.3f}".format(precision))
                # print("recall: {:0.3f}".format(recall))
                print("sensitivity: {:0.3f}".format(sensitivity))
                print("specificity: {:0.3f}".format(specificity))
                print("(sensitivity + specificity) / 2: {:0.3f}".format(ss_2))
                # print("f1_score: {:0.3f}".format(f1_score))
                print()

                result_csv_wf.write('{},{},{:0.1f},{:0.3f},{:0.3f},{:0.3f},{},{},{},{}\n'.format(csv_path, calc_type, threshold, sensitivity, specificity, ss_2, confusion_mat[1][1], confusion_mat[1][2], confusion_mat[2][1], confusion_mat[2][2]))

                if ss_2 > best_ss_2_dict['{}_{}'.format(csv_path, calc_type)][1]:
                    best_ss_2_dict['{}_{}'.format(csv_path, calc_type)] = (threshold, ss_2, confusion_mat)

            output_arr = get_result(data, calc_type=calc_type, return_arr=True)
            output_arr = np.array(output_arr)

            plt.hist(output_arr[1], bins=10, density=True, label="Positive (Cases: {})".format(len(output_arr[1])), alpha=0.5, histtype='stepfilled', color="g")
            plt.hist(output_arr[2], bins=10, density=True, label="Negative (Cases: {})".format(len(output_arr[2])), alpha=0.5, histtype='stepfilled', color="b")
            plt.legend()
            plt.savefig(csv_path.replace('results.csv', "fig_{}.png".format(calc_type)))
            plt.clf()

    result_csv_wf.close()

    for key in sorted(best_ss_2_dict.keys()):
        print(key, ':', best_ss_2_dict[key][0], round(best_ss_2_dict[key][1], 3), best_ss_2_dict[key][2])
