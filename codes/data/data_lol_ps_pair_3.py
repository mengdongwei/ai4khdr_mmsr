import os, sys
sys.path.insert(0, './')
import random
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageFilter
import cv2
from data.dataloader import Dataset
from data.dataloader import DataLoader
import copy
from tools import crash_on_ipy
'''
this is superresolution data loader with choice lr_img from triple(1M, 2M, 4M)
and the hr_img is screen records with usm preocession
'''

class TrainDataset(Dataset):
    def __init__(self, tmp1, tmp2, scale=2, crop_size=128, cfg=None):
        super(TrainDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        self.cfg = cfg

        hr_path_file = '/workspace/nas_mengdongwei/dataset/lol/my_lol_frames_usm_ps_4k/train_crop_paths.txt'
        lr_1M_path_file = '/workspace/nas_mengdongwei/dataset/lol/my_lol_frames_1080_1M/crop_1M_frames_paths.txt'
        lr_2M_path_file = '/workspace/nas_mengdongwei/dataset/lol/my_lol_frames_1080_2M/crop_2M_frames_paths.txt'
        lr_4M_path_file = '/workspace/nas_mengdongwei/dataset/lol/my_lol_frames_1080_4M/crop_4M_frames_paths.txt'

        self.hr_paths = [x.strip() for x in open(hr_path_file, 'r').readlines()]
        self.lr_1M_paths = [x.strip() for x in open(lr_1M_path_file, 'r').readlines()]
        self.lr_2M_paths = [x.strip() for x in open(lr_2M_path_file, 'r').readlines()]
        self.lr_4M_paths = [x.strip() for x in open(lr_4M_path_file, 'r').readlines()]

        assert(len(self.hr_paths) == len(self.lr_1M_paths))
        assert(len(self.hr_paths) == len(self.lr_2M_paths))
        assert(len(self.hr_paths) == len(self.lr_4M_paths))
        self.hr_paths.sort()
        self.lr_1M_paths.sort()
        self.lr_2M_paths.sort()
        self.lr_4M_paths.sort()

        self.num_sample = len(self.hr_paths)

    def _read_pair_img(self, index):
        if random.random() > 0.7:
            lr_img = Image.open(self.lr_4M_paths[index])
        elif random.random() > 0.1:
            lr_img = Image.open(self.lr_2M_paths[index])
        else:
            lr_img = Image.open(self.lr_1M_paths[index])
        if index <= 43308:
            lr_img = lr_img.resize((240, 240))

        hr_img = Image.open(self.hr_paths[index])
        return hr_img, lr_img

    def _lr_add_noise(self, lr):
        if random.random() > 0.7:
            lr_mode = lr.mode
            lr_np = np.array(lr)
            if self.cfg.dataset.add_gaussian_noise:
                if random.random() > 0.5:
                    lr_np = random_noise(lr_np, mode='gaussian', clip=True)
                else:
                    lr_np = random_noise(lr_np, mode='poisson', clip=True)
            if self.cfg.dataset.add_sp_noise:
                lr_np = random_noise(lr_np, mode='s&p', clip=True)

            lr_np = lr_np * 255
            lr_np = lr_np.astype(np.uint8)
            lr = Image.fromarray(lr_np)

        if random.random() > 1:
            lr_np = np.array(lr)
            lr_np = jpeg_compress(lr_np)
            lr_np = lr_np.astype(np.uint8)
            lr = Image.fromarray(lr_np)
        return lr


    def __getitem__(self, index):
        while(True):
            try:
                index = random.randint(43309, self.num_sample)
                hr, lr = self._read_pair_img(index)

                #lr = self._lr_add_noise(lr)
                lr = lr.convert('YCbCr').split()[0]
                lr = np.expand_dims(np.array(lr), -1)

                hr = hr.convert('YCbCr').split()[0]
                hr = np.expand_dims(np.array(hr), -1)

                hr, lr = random_crop(hr, lr, size=self.crop_size, scale=self.scale)
                hr, lr = random_flip_and_rotate(hr, lr)

                cubic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_LINEAR)
                cubic = np.expand_dims(cubic, -1)
                return lr, hr, cubic
            except:
                pass

    def __len__(self):
        return len(self.hr_paths)


class TestDataset(Dataset):
    def __init__(self, hr_path_file, lr_path_file, scale=2, crop_size=None):
        super(TestDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        self.hr_paths_file = '/workspace/nas_mengdongwei/dataset/lol/my_lol_frames_usm_ps_4k/valid_paths.txt'
        self.lr_paths_file = '/workspace/nas_mengdongwei/dataset/lol/my_lol_frames/valid_paths.txt'

        self.hr_paths = [x.strip() for x in open(self.hr_paths_file, 'r').readlines()]
        self.lr_paths = [x.strip() for x in open(self.lr_paths_file, 'r').readlines()]
        assert len(self.hr_paths) == len(self.lr_paths)

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]

        hr = Image.open(hr_path)
        hr = pre_resize(hr)

        lr = Image.open(lr_path)
        lr = lr.resize((hr.size[0]//self.scale, hr.size[1]//self.scale), Image.BILINEAR)

        lr = lr.convert('YCbCr')
        lr = np.expand_dims(np.array(lr.split()[0]), -1)

        hr = hr.convert('YCbCr')
        hr = np.expand_dims(np.array(hr.split()[0]), -1)

        cubic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_LINEAR)
        cubic = np.expand_dims(cubic, -1)
        return lr, hr, cubic

    def __len__(self):
        return len(self.lr_paths)

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
    assert crop_hr.shape == (256, 256, 1)
    assert crop_lr.shape == (128, 128, 1)

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
    from config.config_lol_my_data_ps_pair_3 import cfg
    train_set = TrainDataset(None, None, cfg=cfg)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=64, num_workers=16, shuffle=False, drop_last=True, timeout=0)

    valid_hr_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_valid_hr_paths.txt'
    valid_lr_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_valid_lr_paths.txt'
    test_set = TestDataset(valid_hr_file, valid_hr_file)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=16, shuffle=False, drop_last=True, timeout=0)
    for index, sample in enumerate(test_loader):
        lr, hr, cubic = sample
        print(hr.shape)
        print(lr.shape)
        print(cubic.shape)
