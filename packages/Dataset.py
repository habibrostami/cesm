import os
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

base_transform = transforms.ToTensor()


def is_cc(item):
    return 'CC' in item


def attach(item):
    return item, item.replace('CC', 'MLO')


def check_complete(base_path: str, label: str, cm_cc: str):
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


def full_list(base_path):
    cm_path = os.path.join(base_path, 'CM')
    labels = ['0', '1']
    result = []
    for label in labels:
        dir_path = os.path.join(cm_path, label)
        for name in os.listdir(dir_path):
            if 'MLO' in name:
                continue
            is_complete, paths = check_complete(base_path, label, name)
            if is_complete:
                result.append({'x': paths, 'y': int(label)})
    return result


def list_files(base_path, label, get_types='both'):
    dir_path = os.path.join(base_path, label)
    if get_types == 'both':
        file_names = list(
            map(
                lambda x: (os.path.join(dir_path, x[0]), os.path.join(dir_path, x[1])),
                map(attach, filter(is_cc, os.listdir(dir_path)))
            )
        )
    elif get_types == 'cc':
        file_names = list(map(lambda x: os.path.join(dir_path, x), filter(is_cc, os.listdir(dir_path))))
    elif get_types == 'mlo':
        file_names = list(
            map(lambda x: os.path.join(dir_path, x), filter(lambda x: not is_cc(x), os.listdir(dir_path))))
    else:
        file_names = []
    return list(map(lambda x: {'x': x, 'y': int(label)}, file_names))


def flatten_list(items):
    return [{'x': j, 'y': i['y']} for i in items for j in i['x']]


def read_csv(path: str, cm_or_dm: str, cc_or_mlo: str):
    with open(path, 'r') as file:
        items = file.readlines()
    items = list(map(lambda x: x.split(','), items))
    if cm_or_dm == 'cm':
        items = list(map(lambda x: [x[0], x[1], x[4]], items))
    elif cm_or_dm == 'dm':
        items = list(map(lambda x: [x[2], x[3], x[4]], items))

    if cm_or_dm in ['cm', 'dm'] and cc_or_mlo == 'cc':
        items = list(map(lambda x: [x[0], x[2]], items))
    elif cm_or_dm in ['cm', 'dm'] and cc_or_mlo == 'mlo':
        items = list(map(lambda x: [x[1], x[2]], items))
    elif cm_or_dm in ['cm', 'dm']:
        pass
    elif cc_or_mlo == 'cc':
        items = list(map(lambda x: [x[0], x[2], x[4]], items))
    elif cc_or_mlo == 'mlo':
        items = list(map(lambda x: [x[1], x[3], x[4]], items))
    else:
        pass
    return list(map(lambda x: {'x': x[:-1], 'y': int(x[-1])}, items))


class TwoChannel:
    def __init__(self, base_path, transform=base_transform, device=None):
        self.base_path = base_path
        self.file_paths = list_files(base_path, '0') + list_files(base_path, '1')
        self.transform = transform
        self.device = device

    def __getitem__(self, i):
        # -------------- Get Item --------------
        cc_path, mlo_path = self.file_paths[i]['x']
        label = self.file_paths[i]['y']
        # ------------ Open Images ------------
        cc_image = Image.open(cc_path).convert('L')
        mlo_image = Image.open(mlo_path).convert('L')
        # ----- Create a Two-Channel Image -----
        # -------- Transform and Device --------
        cc_image = self.transform(cc_image)
        mlo_image = self.transform(mlo_image)
        if self.device:
            cc_image = cc_image.to(self.device)
            mlo_image = mlo_image.to(self.device)
        return torch.stack([cc_image.squeeze(0), mlo_image.squeeze(0)]), label

    def __len__(self):
        return len(self.file_paths)


class OneImage(Dataset):
    def __init__(self, base_path, cm_or_dm, cc_or_mlo, transform=base_transform, device=None):
        self.file_paths = read_csv(base_path, cm_or_dm, cc_or_mlo)
        self.transform = transform
        self.device = device
        self.first = True

    def __getitem__(self, i):
        # -------------- Get Item --------------
        image_path = self.file_paths[i]['x']
        # print(image_path)
        label = self.file_paths[i]['y']
        # ------------ Open Images ------------
        image = Image.open(image_path[0])
        # ------------ Show Images ------------
        if self.first:
            self.first = not self.first
            image.show()
        # -------- Transform and Device --------
        result = self.transform(image)
        if self.device:
            result = result.to(self.device)
        result = (result - torch.mean(result)) / torch.std(result)
        return result, label

    def __len__(self):
        return len(self.file_paths)


class FourImages(Dataset):
    def __init__(self, base_path, transform=base_transform, device=None, flat=False):
        self.transform = transform
        self.file_paths = read_csv(base_path, 'both', 'both')
        if flat:
            self.file_paths = flatten_list(self.file_paths)
        self.device = device
        self.first = True

    def __getitem__(self, i):
        # -------------- Get Item --------------
        image_paths = self.file_paths[i]['x']
        label = self.file_paths[i]['y']
        # ------------ Open Images ------------
        image0 = Image.open(image_paths[0])
        image1 = Image.open(image_paths[1])
        image2 = Image.open(image_paths[2])
        image3 = Image.open(image_paths[3])
        # ------------ Show Images ------------
        if self.first:
            self.first = not self.first
            image0.show()
            image1.show()
            image2.show()
            image3.show()
        # -------- Transform and Device --------
        image0 = self.transform(image0)
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        image3 = self.transform(image3)
        if self.device:
            image0 = image0.to(self.device)
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            image3 = image3.to(self.device)
        return image0, image1, image2, image3, label

    def __len__(self):
        return len(self.file_paths)


class FourImagesPreSegment(Dataset):
    def __init__(self, base_path, mask_dir, transform=base_transform, device=None, flat=False):
        self.transform = transform
        self.file_paths = read_csv(base_path, 'both', 'both')
        if flat:
            self.file_paths = flatten_list(self.file_paths)
        self.device = device
        self.first = True
        self.mask_dir = mask_dir

    def __getitem__(self, i):
        # -------------- Get Item --------------
        image_paths = self.file_paths[i]['x']
        label = self.file_paths[i]['y']
        mask_paths = [os.path.join(self.mask_dir, f'{image_path.split("/")[-1]}.png') for image_path in image_paths]
        # ------------ Open Images ------------
        image0 = Image.open(image_paths[0])
        image1 = Image.open(image_paths[1])
        image2 = Image.open(image_paths[2])
        image3 = Image.open(image_paths[3])
        masks = []
        for image, mask_path in zip([image0, image1, image2, image3], mask_paths):
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
            else:
                print('mask does not exist!', end='')
                mask = Image.new("L", image.size, 0)
            masks.append(mask)

        # ------------ Show Images ------------
        if self.first:
            self.first = not self.first
            image0.show()
            image1.show()
            image2.show()
            image3.show()
            masks[0].show()
            masks[1].show()
            masks[2].show()
            masks[3].show()
        # -------- Transform and Device --------
        image0 = self.transform(image0)
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        image3 = self.transform(image3)
        masks = [self.transform(mask) for mask in masks]
        for i, mask in enumerate(masks):
            mask[mask > 0.5] = 1
            mask[mask != 1] = 0
            if self.device:
                masks[i] = mask.to(self.device)
        if self.device:
            image0 = image0.to(self.device)
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            image3 = image3.to(self.device)
        return image0, image1, image2, image3, *masks, label

    def __len__(self):
        return len(self.file_paths)


class TwoImages(Dataset):
    def __init__(self, base_path, join_type, transform=base_transform, device=None, flat=False):
        self.transform = transform
        if join_type in ['cc', 'mlo']:
            self.file_paths = read_csv(base_path, 'both', join_type)
        else:
            self.file_paths = read_csv(base_path, join_type, 'both')
        if flat:
            self.file_paths = flatten_list(self.file_paths)
        self.device = device
        self.first = True

    def __getitem__(self, i):
        # -------------- Get Item --------------
        image_paths = self.file_paths[i]['x']
        label = self.file_paths[i]['y']
        # ------------ Open Images ------------
        image0 = Image.open(image_paths[0])
        image1 = Image.open(image_paths[1])
        # ------------ Show Images ------------
        if self.first:
            self.first = not self.first
            image0.show()
            image1.show()
        # -------- Transform and Device --------
        image0 = self.transform(image0)
        image1 = self.transform(image1)
        if self.device:
            image0 = image0.to(self.device)
            image1 = image1.to(self.device)
        return image0, image1, label

    def __len__(self):
        return len(self.file_paths)


class SegmentDataset(Dataset):
    def __init__(self, base_path, cm_or_dm, cc_or_mlo, mask_dir, transform=base_transform, device=None):
        assert cm_or_dm != 'both', "Cannot get both CM and DM. Choose one..."
        assert cc_or_mlo != 'both', "Cannot get both CC and MLO. Choose one..."
        self.mask_dir = mask_dir
        self.file_paths = read_csv(base_path, cm_or_dm, cc_or_mlo)
        self.transform = transform
        self.device = device
        self.first = True

    def __getitem__(self, i):
        # -------------- Get Item --------------
        image_path = self.file_paths[i]['x'][0]
        mask_path = os.path.join(self.mask_dir, f'{image_path.split("/")[-1]}.png')
        # print(mask_path)
        # ------------ Open Images ------------
        image = Image.open(image_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            print('mask does not exist!', end='')
            mask = Image.new("L", image.size, 0)

        # ------------ Show Images ------------
        if self.first:
            self.first = not self.first
            image.show()
            mask.show()
        # -------- Transform and Device --------
        result = self.transform(image)
        mask = self.transform(mask)
        mask[mask > 0.5] = 1
        mask[mask != 1] = 0
        mask = mask.long()
        if self.device:
            result = result.to(self.device)
            mask = mask.to(self.device)
        result = (result - torch.mean(result)) / torch.std(result)
        mask = mask.squeeze(0)
        return result, mask

    def __len__(self):
        return len(self.file_paths)


class PreSegmentDataset(Dataset):
    def __init__(self, base_path, cm_or_dm, cc_or_mlo, mask_dir, transform=base_transform, device=None):
        assert cm_or_dm != 'both', "Cannot get both CM and DM. Choose one..."
        assert cc_or_mlo != 'both', "Cannot get both CC and MLO. Choose one..."
        self.mask_dir = mask_dir
        self.file_paths = read_csv(base_path, cm_or_dm, cc_or_mlo)
        self.transform = transform
        self.device = device
        self.first = True

    def __getitem__(self, i):
        # -------------- Get Item --------------
        image_path = self.file_paths[i]['x'][0]
        label = self.file_paths[i]['y']
        mask_path = os.path.join(self.mask_dir, f'{image_path.split("/")[-1]}.png')
        # print(mask_path)
        # ------------ Open Images ------------
        image = Image.open(image_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            print('mask does not exist!', end='')
            mask = Image.new("L", image.size, 0)

        # ------------ Show Images ------------
        if self.first:
            self.first = not self.first
            image.show()
            mask.show()
        # -------- Transform and Device --------
        result = self.transform(image)
        mask = self.transform(mask)
        mask[mask > 0.5] = 1
        mask[mask != 1] = 0
        if self.device:
            result = result.to(self.device)
            mask = mask.to(self.device)
        result = (result - torch.mean(result)) / torch.std(result)
        return result, mask, label

    def __len__(self):
        return len(self.file_paths)


class SegmentTwoImage(Dataset):
    def __init__(self, base_path, join_type, mask_dir, transform=base_transform, device=None):
        self.transform = transform
        self.mask_dir = mask_dir
        if join_type in ['cc', 'mlo']:
            self.file_paths = read_csv(base_path, 'both', join_type)
        else:
            self.file_paths = read_csv(base_path, join_type, 'both')
        self.file_paths = flatten_list(self.file_paths)
        self.device = device
        self.first = True

    def __getitem__(self, i):
        # -------------- Get Item --------------
        # print(self.file_paths[i]['x'])
        image_path = self.file_paths[i]['x']
        mask_path = os.path.join(self.mask_dir, f'{image_path.split("/")[-1]}.png')
        # ------------ Open Images ------------
        # print(image_path)
        image = Image.open(image_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            print('mask does not exist!', end='')
            mask = Image.new("L", image.size, 0)
        # ------------ Show Images ------------

        if self.first:
            self.first = not self.first
            image.show()
            mask.show()
        # -------- Transform and Device --------
        result = self.transform(image)
        mask = self.transform(mask)
        mask[mask > 0.5] = 1
        mask[mask != 1] = 0
        mask = mask.long()
        if self.device:
            result = result.to(self.device)
            mask = mask.to(self.device)
        result = (result - torch.mean(result)) / torch.std(result)
        return result, mask

    def __len__(self):
        return len(self.file_paths)
