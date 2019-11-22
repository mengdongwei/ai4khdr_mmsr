import os, sys
sys.path.insert(0, './')
import random
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageFilter
import torch
import cv2
from data.dataloader_x2 import Dataset
from data.dataloader_x2 import DataLoader
import copy
from tools import crash_on_ipy
'''
this is superresolution data loader with choice lr_img from triple(06M, 08M, 12M)
and the hr_img is screen records with usm preocession
'''

class TrainDataset(Dataset):
    def __init__(self, scale=2, crop_size=384):
        super(TrainDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        ### outdoor data ======================================================
        hr_path_file = '/workspace/nas_mengdongwei/dataset/outdoor_v2/train/path_train_1080_usm_crop.txt'
        lr_12M_path_file = '/workspace/nas_mengdongwei/dataset/outdoor_v2/train/path_train_540_12_crop.txt'
        self.hr_paths = [x.strip() for x in open(hr_path_file, 'r').readlines()]
        self.lr_12M_paths = [x.strip() for x in open(lr_12M_path_file, 'r').readlines()]
        assert(len(self.hr_paths) == len(self.lr_12M_paths))
        self.hr_paths.sort()
        self.lr_12M_paths.sort()
        self.outdoor_len = len(self.hr_paths)
        print('num sample of outdoor %04d' %(self.outdoor_len))

        ### outdoor_part2 ======================================================
        outdoor_part2_train_hr_path_file = '/workspace/nas_mengdongwei/dataset/outdoor_v2/train/path_train_1080_usm_part2_crop.txt'
        outdoor_part2_train_lr_path_file = '/workspace/nas_mengdongwei/dataset/outdoor_v2/train/path_train_540_12_part2_crop.txt'

        outdoor_part2_train_hr_paths = [x.strip() for x in open(outdoor_part2_train_hr_path_file, 'r').readlines()]
        outdoor_part2_train_lr_paths = [x.strip() for x in open(outdoor_part2_train_lr_path_file, 'r').readlines()]

        self.outdoor_part2_hr_paths = outdoor_part2_train_hr_paths
        self.outdoor_part2_lr_paths = outdoor_part2_train_lr_paths
        self.outdoor_part2_hr_paths.sort()
        self.outdoor_part2_lr_paths.sort()
        assert(len(self.outdoor_part2_hr_paths) == len(self.outdoor_part2_lr_paths))
        self.outdoor_part2_len = len(self.outdoor_part2_hr_paths)
        print('num sample of outdoor_part2 %04d' %(self.outdoor_part2_len))


        self.hr_paths = self.hr_paths + self.outdoor_part2_hr_paths
        self.lr_12M_paths = self.lr_12M_paths + self.outdoor_part2_lr_paths


    def __len__(self):
        return len(self.hr_paths)

    def _read_pair_img(self, index, dataset_flag):
        lr_img = Image.open(self.lr_12M_paths[index])
        hr_img = Image.open(self.hr_paths[index])
        return hr_img, lr_img

    def __getitem__(self, index):
        hr, lr = self._read_pair_img(index, None)

        lr = lr.convert('YCbCr').split()[0]
        lr = np.expand_dims(np.array(lr), -1).astype(np.float32)

        hr = hr.convert('YCbCr').split()[0]
        hr = np.expand_dims(np.array(hr), -1).astype(np.float32)

        hr, lr = random_crop(hr, lr, size=self.crop_size, scale=self.scale)
        hr, lr = random_flip_and_rotate(hr, lr)

        lrx2 = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_LINEAR)
        lrx2 = np.expand_dims(lrx2, -1)

        # H W C -> C H W
        lr =  np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))
        lr = torch.from_numpy(lr).float()
        hr = np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))
        hr = torch.from_numpy(hr).float()
        lrx2 =  np.ascontiguousarray(np.transpose(lrx2, (2, 0, 1)))
        lrx2 = torch.from_numpy(lrx2).float()
        data = {}
        data['LQ'] = lr
        data['GT'] = hr
        data['LQX2'] = lrx2
        return data


class ValidDataset(Dataset):
    def __init__(self, scale=2, crop_size=None):
        super(ValidDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        hr_paths_file = '/workspace/nas_mengdongwei/dataset/outdoor_v2/valid/valid_path_1080_usm.txt'
        lr_12M_path_file = '/workspace/nas_mengdongwei/dataset/outdoor_v2/valid/valid_path_540_12.txt'

        self.hr_paths = [x.strip() for x in open(hr_paths_file, 'r').readlines()]
        self.lr_12M_paths = [x.strip() for x in open(lr_12M_path_file, 'r').readlines()]
        assert(len(self.hr_paths) == len(self.lr_12M_paths))
        self.hr_paths.sort()
        self.lr_12M_paths.sort()

        self.lr_paths = self.lr_12M_paths
        self.outdoor_len = len(self.hr_paths)

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]

        hr = Image.open(hr_path)
        hr = pre_resize(hr)

        lr = Image.open(lr_path)
        lr = lr.resize((hr.size[0]//self.scale, hr.size[1]//self.scale), Image.BILINEAR)

        lr = lr.convert('YCbCr')
        lr = np.expand_dims(np.array(lr.split()[0]), -1).astype(np.float32)

        hr = hr.convert('YCbCr')
        hr = np.expand_dims(np.array(hr.split()[0]), -1).astype(np.float32)

        lrx2 = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_LINEAR)
        lrx2 = np.expand_dims(lrx2, -1)

        # H W C -> C H W
        lr =  np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))
        lr = torch.from_numpy(lr).float()
        hr = np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))
        hr = torch.from_numpy(hr).float()
        lrx2 =  np.ascontiguousarray(np.transpose(lrx2, (2, 0, 1)))
        lrx2 = torch.from_numpy(lrx2).float()
        data = {}
        data['LQ'] = lr
        data['GT'] = hr
        data['LQX2'] = lrx2
        return data

    def __len__(self):
        return len(self.lr_paths)


def cv2_read_img(path):
    return cv2.imread(path)

def pil_read_img(path):
    return Image.open(path)

def cv_2_pil(img):
    return Image.fromarray(img)

def pre_resize(hr):
    hr_w, hr_h = hr.size
    hr_w = hr_w // 2 * 2
    hr_h = hr_h // 2 * 2
    hr = hr.resize((hr_w, hr_h))
    return hr


def jpeg_compress(img):
    quality = random.randint(30, 100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    crop_lr = lr[y: y + size, x: x + size].copy()
    crop_hr = hr[hy: hy + hsize, hx: hx + hsize].copy()
    assert crop_hr.shape == (size*2, size*2, 1)
    assert crop_lr.shape == (size, size, 1)

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    return im1.copy(), im2.copy()

if __name__ == '__main__':
    train_set = TrainDataset(crop_size=192)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=64, num_workers=16, shuffle=False, drop_last=True, timeout=0)

    test_set = ValidDataset()
    test_loader = DataLoader(test_set, batch_size=1, num_workers=16, shuffle=False, drop_last=True, timeout=0)
    for index, sample in enumerate(test_set):
        if isinstance(sample, tuple):
            lr, hr, lrx2 = sample[0], sample[1], sample[2]
        elif isinstance(sample, dict):
            lr, hr, lrx2 = sample['LQ'], sample['GT'], sample['LQX2']
        print(index)
        print(hr.shape)
        print(lr.shape)
