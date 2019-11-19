import os, sys
import torch
from datetime import datetime
sys.path.insert(0, './')
import random
import numpy as np
from PIL import Image, ImageFilter
import cv2
import copy
from utils import crash_on_ipy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from utils import util
from models import create_model
from utils import crash_on_ipy


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

class Test():
    def __init__(self, n_frames=5, cfg=None):
        super(Test, self).__init__()

        self.cfg = cfg
        self.n_frames = n_frames
        lr_root = '/workspace/nas_mengdongwei/dataset/AI4KHDR/test/test_540p_frame_list_paths.txt'

        self.lr_paths_list, self.num_lr_video, num_lr_img_per_video, self.video_names = get_frames_path(lr_root)
        total_lr_img = np.sum(num_lr_img_per_video)
        self.num_video = self.num_lr_video
        self.num_img_per_video = num_lr_img_per_video

        seed = random.randint(1, 10000)
        util.set_random_seed(seed)
        torch.backends.cudnn.benchmark = True

    def __len__(self):
        return total_lr_img

    def load_model(self):
        opt_path = './options/test/test_EDVR_M_AI4KHDR.yml'
        opt = option.parse(opt_path, is_train=False)
        opt['dist'] = False
        rank = -1
        opt = option.dict_to_nonedict(opt)
        torch.backends.cudnn.benchmark = True
        #### create model
        model = create_model(opt)
        return model

    def predict(self, model, lrs, gt=None):
        data = (lrs, gt)

        model.feed_data(data, need_GT=(gt is not None))
        model.test()
        visuals = model.get_current_visuals(nedd_GT=(gt is not None))
        sr_img = util.tensor2img(visuals['rlt'])  # uint8
        if gt:
            gt_img = util.tensor2img(visuals['GT'])  # uint8
        else:
            gt_img = None

        return sr_img, gt_img


    def eval(self):
        model = self.load_model()

        # mkdir test save dir
        root_dir = 'ai4khdr_test_edvr_m'
        save_dir = os.path.join(root_dir, get_timestamp())
        makedir(save_dir)

        for ii in range(self.num_video):
            video_name = self.video_names[ii]
            video_result_dir = os.path.join(save_dir, video_name)

            lrs = [np.array(Image.open(self.lr_paths_list[ii][jj])) for jj in range(self.num_img_per_video[ii])]
            num_frames = len(lrs)
            lrs = lrs[0] * self.num_frames // 2 + lrs + lrs[-1] * self.num_frames // 2 # padding two frame at the head and the tail of video

            for kk in range(0, num_frames):
                sample = lrs[kk:kk+self.num_frames]
                sample = np.array(sample)

                # N H W C -> N C H W
                sample =  np.ascontiguousarray(np.transpose(sample, (0, 3, 1, 2)))
                sample = sample[np.new_aixs, :, :, :, :]
                print(sample.shape, 'sample')
                sample = torch.from_numpy(sample).float()
                sr, gt = self.predict(model, sample)
                print(sr.shape, 'sr')

                save_path = os.path.join(video_result_dir, '%04d.png'%(kk + self.num_frames//2))
                print(save_path, 'save path')
                util.save_img(sr, save_path)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_frames_path(root_file):
    path_list = []
    level_1_list = open(root_file, 'r').readlines()
    level_1_list.sort()
    video_names = []
    total_img_per_video = []
    total_video = len(level_1_list)
    for video_list in level_1_list:
        video_names.append(os.path.basename(video_list.strip()).split('_img')[0])
        level_2_list = open(video_list.strip(), 'r').readlines()
        level_2_list.sort()
        one_video_frames_path = [x.strip() for x in level_2_list]
        path_list.append(one_video_frames_path)
        total_img_per_video.append(len(one_video_frames_path))
    return path_list, total_video,  total_img_per_video, video_names

if __name__ == '__main__':
    init_dist()
    test = Test()
    test.eval()
