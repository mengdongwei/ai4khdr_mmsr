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
    def __init__(self, scale=2, crop_size=128, cfg=None):
        super(TrainDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        self.cfg = cfg

        hr_path_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_train_hr_crop_paths.txt'
        lr_path_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_train_lr_crop_paths.txt'

        self.hr_paths = [x.strip() for x in open(hr_path_file, 'r').readlines()]
        self.lr_paths = [x.strip() for x in open(lr_path_file, 'r').readlines()]

        assert(len(self.hr_paths) == len(self.lr_paths))
        self.hr_paths.sort()
        self.lr_paths.sort()

        self.num_sample = len(self.hr_paths)

    def _read_pair_img(self, index):
        lr_img = Image.open(self.lr_paths[index])
        hr_img = Image.open(self.hr_paths[index])
        return hr_img, lr_img

    def _lr_add_noise(self, lr):
        if random.random() > 0.5:
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

        if random.random() > 0.3:
            lr_np = np.array(lr)
            lr_np = jpeg_compress(lr_np)
            lr_np = lr_np.astype(np.uint8)
            lr = Image.fromarray(lr_np)
        return lr

    def resize(self, image, size):
        resamples = [Image.NEAREST, Image.BILINEAR, Image.HAMMING, \
                     Image.BICUBIC, Image.LANCZOS]
        resample = random.choice(resamples)
        return image.resize(size, resample=resample)

    def gaussianblur(self, image, radius=2):
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def medianfilter(self, image, size=3):
        return image.filter(ImageFilter.MedianFilter(size=size))

    def downsampling(self, image):
        resize = (image.size[0]//self.scale, image.size[1]//self.scale)
        #hidden_scale = random.uniform(1, 1.5)
        #hidden_resize = (int(resize[0]/hidden_scale), int(resize[1]/hidden_scale))
        #radius = random.uniform(1, 3)
        #image = self.gaussianblur(image, radius)
        #image = self.resize(image, hidden_resize)
        image = self.resize(image, resize)
        return image

    def __getitem__(self, index):
        #index = random.randint(0, self.num_sample-1)
        hr, lr = self._read_pair_img(index)
        lr = self.downsampling(hr)

        lr = self._lr_add_noise(lr)
        lr = lr.convert('YCbCr').split()[0]
        lr = np.expand_dims(np.array(lr), -1)

        hr = hr.convert('YCbCr').split()[0]
        hr = np.expand_dims(np.array(hr), -1)

        hr, lr = random_crop(hr, lr, size=self.crop_size, scale=self.scale)
        hr, lr = random_flip_and_rotate(hr, lr)

        cubic = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_LINEAR)
        cubic = np.expand_dims(cubic, -1)
        assert hr.shape == (256, 256, 1)
        assert cubic.shape == (256, 256, 1)
        assert lr.shape == (128, 128, 1)
        return lr, hr, cubic

    def __len__(self):
        return len(self.hr_paths)

class TestDataset(Dataset):
    def __init__(self, scale=2, crop_size=None):
        super(TestDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        self.hr_path_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_valid_hr_paths.txt'
        self.lr_path_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_valid_lr_paths.txt'

        self.hr_paths = [x.strip() for x in open(self.hr_path_file, 'r').readlines()]
        self.lr_paths = [x.strip() for x in open(self.lr_path_file, 'r').readlines()]
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
    from config.config_outdoor_esrgan import cfg
    train_set = TrainDataset(2, 128, cfg=cfg)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=64, num_workers=16, shuffle=False, drop_last=True, timeout=0)

    valid_hr_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_valid_hr_paths.txt'
    valid_lr_file = '/workspace/nas_mengdongwei/dataset/div2k/div2k_valid_lr_paths.txt'
    test_set = TrainDataset(valid_hr_file, valid_hr_file)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=16, shuffle=False, drop_last=True, timeout=0)
    for index, sample in enumerate(train_set):
        lr, hr, cubic = sample
        if index % 100 == 0:
            print(index)
