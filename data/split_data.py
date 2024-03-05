import os
import random
from tqdm import tqdm

import informations as info

random.seed(0)
DATA_TYPE = list[tuple[str, str, str]]
 
def get_data(datapath: str,
             class_labels: list[str],
             backgrounds: list[str],
             only_retouche: bool=True,
             ) -> list[tuple[str, str, str]]:
    """ get data
    list of tuple that containt image_path, label, background"""
    all_image_paths = []
    all_image_labels = []
    all_image_backgrounds = []

    for class_label in class_labels:
        for root, _, files in tqdm(os.walk(os.path.join(datapath, class_label)), desc='walk in data'):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.JPG')):  # add or remove file extensions as needed
                    path = os.path.join(root, file)
                    if (not only_retouche) or "Retouches" in path:  # on ne prend pas les images non retouchées
                        all_image_paths.append(path)
                        all_image_labels.append(class_label)
                        for bg in backgrounds:
                            if bg in path.lower():
                                all_image_backgrounds.append(bg)

    c = list(zip(all_image_paths, all_image_labels, all_image_backgrounds))
    random.shuffle(c)
    print(f'get {len(c)} data')
    return c


def split_data(data: DATA_TYPE,
               train_rate: float=0.8,
               val_rate: float=0.1,
               test_rate: float=0.1
               ) -> tuple[DATA_TYPE, DATA_TYPE, DATA_TYPE]:
    """ split the data in 3 parts according rate proportion """
    if train_rate + val_rate + test_rate != 1:
        raise ValueError(f'train + val + test rate must be equal to 1,',
                         f'but is equal to {train_rate + val_rate + test_rate}.')

    n = len(data)
    split_1 = int(n * train_rate)
    split_2 = int(n * (train_rate + val_rate))

    train_data = data[:split_1]
    val_data = data[split_1 : split_2]
    test_data = data[split_2:]

    return train_data, val_data, test_data


def save_data(data: DATA_TYPE, mode: str, datapath: str) -> None:
    """ save data in csv in datapath with the name {mode}_item.csv"""
    output = ''
    for i, (path, label, background) in enumerate(data):
            output += f'{i},{path},{label},{background}\n'
    
    dstpath = os.path.join(datapath, f'{mode}_item.csv')
    with open(file=dstpath, mode='w', encoding='utf8') as f:
        f.write(output)
        f.close()
    print(f'data save in {dstpath}')
        

if __name__ == '__main__':
    data = get_data(datapath=info.DATAPATH,
                    class_labels=info.LABELS,
                    backgrounds=info.BACKGROUND,
                    only_retouche='retouche' in info.DATAPATH)
    print(data)
    print(len(data))
    MODE = ['train', 'val', 'test']
    data_split = split_data(data)

    for i in range(3):
        save_data(data=data_split[i],
                  mode=MODE[i],
                  datapath='data')


    