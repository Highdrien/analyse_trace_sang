import os

import informations as info


def get_distribution(mode: str,
                     datapath: str=os.path.join('data', 'data_labo')
                     ) -> list[float]:
    datapath = os.path.join(datapath, f'{mode}_128')
    labels_distribution = []
    for label in info.LABELS:
        num_data = 0
        for bg in info.BACKGROUND:
            num_data += len(os.listdir(os.path.join(datapath, label, bg)))
        labels_distribution.append(num_data)

    N = sum(labels_distribution)
    for i in range(len(labels_distribution)):
        labels_distribution[i] /= N
    
    return labels_distribution


if __name__ == '__main__':
    train = get_distribution(mode='train')
    val = get_distribution(mode='val')
    test = get_distribution(mode='test')

    print('label, train, val, test')
    for i in range(len(info.LABELS)):
        print(f'{i + 1:2}: {train[i]:.2f} {val[i]:.2f} {test[i]:.2f}')