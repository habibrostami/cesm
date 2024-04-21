import os
import shutil
from math import floor
from typing import List, Dict, Optional

base_path = './data'
save_path = './data'


def check_complete(label: str, cm_cc: str):
    cm_mlo = cm_cc.replace('CC', 'MLO')
    dm_cc = cm_cc.replace('CM', 'DM')
    dm_mlo = cm_mlo.replace('CM', 'DM')
    names = [os.path.join(base_path, 'CM', label, cm_cc),
             os.path.join(base_path, 'CM', label, cm_mlo),
             os.path.join(base_path, 'DM', label, dm_cc),
             os.path.join(base_path, 'DM', label, dm_mlo),
             ]

    if all(map(lambda x: os.path.isfile(x), names)):
        return True, names
    return False, []


def full_list():
    cm_path = os.path.join(base_path, 'CM')
    labels = ['0', '1']
    result = {'0': [], '1': []}
    for label in labels:
        dir_path = os.path.join(cm_path, label)
        for name in os.listdir(dir_path):
            if 'MLO' in name:
                continue
            is_complete, paths = check_complete(label, name)
            if is_complete:
                result[label].append(paths)
    return result


def splitter(data: Dict[str, List[str]], sizes: List[float], names: Optional[List[str]]):
    assert 0.99 <= sum(sizes) <= 1.0, "Sizes are not covering exactly all data. try numbers that sum to 1"
    if names:
        assert len(sizes) == len(names), "names and sizes must be the same size"
    else:
        names = [str(i) for i in range(1, len(sizes) + 1)]
    result = {}
    for label in data:
        all_items = data[label]
        n_all = len(all_items)
        n_splits = list(map(lambda x: floor(x * n_all), sizes))
        print([f'{i}: {j}' for i, j in zip(names, n_splits)])
        n_splits[-1] += n_all - sum(n_splits)
        idx = [0] + [x + sum(n_splits[:i]) for i, x in enumerate(n_splits)]
        splits = [all_items[idx[i]: idx[i + 1]] for i in range(len(idx) - 1)]
        result[label] = {x: y for x, y in zip(names, splits)}
    return result


def main():
    result = splitter(full_list(), [0.7, 0.15, 0.15], ['train', 'validation', 'test'])
    for label in result:
        data = result[label]
        for split in data:
            with open(f'./{split}_data.csv', 'a+') as file:
                for item in data[split]:
                    listed = [i.split('/') for i in item]
                    for x in listed:
                        x.insert(2, split)
                    mapper = {i: ['/'.join(j[:-1]), j[-1]] for i, j in zip(item, listed)}
                    for path in mapper:
                        save_dir, save_name = mapper[path]
                        os.makedirs(save_dir, exist_ok=True)
                        shutil.copy(path, os.path.join(save_dir, save_name))
                        file.write(os.path.join(save_dir, save_name))
                        file.write(',')
                    file.write(label + '\n')


if __name__ == '__main__':
    main()
