import os, sys
import torch
sys.path.insert(0, './')
import random
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageFilter
import cv2
from data_m.dataloader import Dataset
from data_m.dataloader import DataLoader
import copy
from utils import crash_on_ipy

class TrainDataset(Dataset):
    def __init__(self, scale=4, crop_size=224, n_frames=5, cfg=None):
        super(TrainDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        self.cfg = cfg
        self.n_frames = n_frames

        hr_root = '/workspace/datasets/AI_4K_HDR/train/train_4k_frame_list_file.txt'
        lr_root = '/workspace/datasets/AI_4K_HDR/train/train_540p_frame_list_file.txt'

        self.hr_paths_list, self.num_hr_video, num_hr_img_per_video = get_frames_path(hr_root)
        self.lr_paths_list, self.num_lr_video, num_lr_img_per_video = get_frames_path(lr_root)
        total_hr_img = np.sum(num_hr_img_per_video)
        total_lr_img = np.sum(num_lr_img_per_video)
        assert total_hr_img == total_lr_img
        self.num_sample = total_hr_img - self.n_frames * self.num_hr_video
        self.num_video = self.num_hr_video
        self.num_img_per_video = num_hr_img_per_video

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        video_index = random.randint(0, self.num_video-1)
        frame_index = random.randint(self.n_frames//2 + 1, self.num_img_per_video[video_index] - (self.n_frames//2 + 1)-1)
        hr = np.array(Image.open(self.hr_paths_list[video_index][frame_index]))
        lrs = []
        for xx in range(- (self.n_frames//2), self.n_frames//2+1, 1):
            lr = np.array(Image.open(self.lr_paths_list[video_index][frame_index + xx]))
            lrs.append(lr)
        lrs = np.array(lrs)
        hr, lrs = random_crop(hr, lrs, size=self.crop_size, scale=self.scale)
        hr, lrs = random_flip_and_rotate(hr, lrs)

        # N H W C -> N C H W
        lrs =  np.ascontiguousarray(np.transpose(lrs/255, (0, 3, 1, 2)))
        lrs = torch.from_numpy(lrs).float()
        # H W C -> C H W
        hr = np.ascontiguousarray(np.transpose(hr/255, (2, 0, 1)))
        hr = torch.from_numpy(hr).float()
        return lrs, hr


class ValidDataset(Dataset):
    def __init__(self, scale=4, n_frames=5, crop_size=None):
        super(ValidDataset, self).__init__()

        self.crop_size = crop_size
        self.scale = scale
        self.n_frames = n_frames
        hr_root = '/workspace/nas_mengdongwei/dataset/AI4KHDR/valid/valid_4k_frame_list_file.txt'
        lr_root = '/workspace/nas_mengdongwei/dataset/AI4KHDR/valid/valid_540p_frame_list_file.txt'

        self.hr_paths_list, self.num_hr_video, num_hr_img_per_video = get_frames_path(hr_root)
        self.lr_paths_list, self.num_lr_video, num_lr_img_per_video = get_frames_path(lr_root)
        total_hr_img = np.sum(num_hr_img_per_video)
        total_lr_img = np.sum(num_lr_img_per_video)
        assert total_hr_img == total_lr_img
        self.num_sample = total_hr_img - n_frames * self.num_hr_video
        self.num_video = self.num_hr_video
        self.num_img_per_video = num_hr_img_per_video

    def __len__(self):
        return 600 #self.num_sample

    def __getitem__(self, index):
        video_index = index // 100 #random.randint(0, self.num_video-1)
        frame_index = index % 100
        if frame_index < 2:
            frame_index == 2
        elif frame_index > 97:
            frame_index = 97
        #frame_index = random.randint(self.n_frames//2 + 1, self.num_img_per_video[video_index] - (self.n_frames//2 + 1) - 1)

        hr = np.array(Image.open(self.hr_paths_list[video_index][frame_index]))
        lrs = []
        for xx in range(- (self.n_frames//2), self.n_frames//2+1, 1):
            lr = np.array(Image.open(self.lr_paths_list[video_index][frame_index + xx]))
            lrs.append(lr)
        lrs = np.array(lrs)

        # N H W C -> N C H W
        lrs =  np.ascontiguousarray(np.transpose(lrs/255, (0, 3, 1, 2)))
        lrs = torch.from_numpy(lrs).float()
        # H W C -> C H W
        hr = np.ascontiguousarray(np.transpose(hr/255, (2, 0, 1)))
        hr = torch.from_numpy(hr).float()
        return lrs, hr

def get_frames_path(root_file):
    path_list = []
    level_1_list = open(root_file, 'r').readlines()
    level_1_list.sort()
    total_img_per_video = []
    total_video = len(level_1_list)
    for video_list in level_1_list:
        level_2_list = open(video_list.strip(), 'r').readlines()
        level_2_list.sort()
        one_video_frames_path = [x.strip() for x in level_2_list]
        path_list.append(one_video_frames_path)
        total_img_per_video.append(len(one_video_frames_path))
    return path_list, total_video,  total_img_per_video


def jpeg_compress(img):
    quality = random.randint(30, 100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def random_crop(hr, lrs, size, scale):
    h, w = lrs.shape[1:3]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    crop_lr = lrs[:, y: y + size, x: x + size, :].copy()
    crop_hr = hr[hy: hy + hsize, hx: hx + hsize, :].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(hr, lrs):
    if random.random() < 0.5:
        hr = np.flipud(hr)
        lrs = np.flipud(lrs)

    if random.random() < 0.5:
        hr = np.fliplr(hr)
        lrs = np.fliplr(lrs)

    angle = random.choice([0, 1, 2, 3])
    hr = np.rot90(hr, angle)
    lrs = np.rot90(lrs, angle, (1, 2))

    return hr.copy(), lrs.copy()

if __name__ == '__main__':
    train_set = TrainDataset()
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=4, num_workers=16, shuffle=False, drop_last=True, timeout=0)

    test_set = ValidDataset()
    test_loader = DataLoader(test_set, batch_size=1, num_workers=16, shuffle=False, drop_last=True, timeout=0)
    for index, sample in enumerate(train_set):
        lr, hr, lrx2 = sample
        print('hr', hr.shape)
        print('lr', lr.shape)
        print('x2', lrx2.shape)

