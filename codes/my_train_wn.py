import os
import math
import argparse
import random
import logging
logging.disable(logging.WARNING)

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.data_sampler import DistIterSampler
import options.options as option
from utils import util
from data_m.data_hdr import TrainDataset, ValidDataset
from data import create_dataloader
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


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    opt['dist'] = True
    init_dist()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    #if resume_state is None:
    #    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
    #    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
    #                 and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        if version >= 1.1:  # PyTorch 1.1
            from torch.utils.tensorboard import SummaryWriter
        else:
            logger.info(
                'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
            from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 1 #200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = TrainDataset()
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
            total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = ValidDataset()
            train_size = int(math.ceil(len(val_set) / dataset_opt['batch_size']))
            val_sampler = DistIterSampler(val_set, world_size, rank, dataset_ratio)
            val_loader = create_dataloader(val_set, dataset_opt, opt, val_sampler)
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.3f} '.format(k, v)
                logger.info(message)
                print(message)
            #### save models and training states
            if current_step % 1000 == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

                #### validation # multi-GPU testing
                psnr_rlt = []  # with border and center frames
                for _, val_data in enumerate(val_loader):
                    model.feed_data(val_data)
                    model.test()
                    # calculate PSNR
                    psnr_rlt.append( torch.mean(model.get_psnr()).item())

                psnr_total_avg = np.array(psnr_rlt).mean()
                log_s = '# Validation # PSNR: {:.4f}:'.format(psnr_total_avg)
                logger.info(log_s)
                print(log_s)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')
    tb_logger.close()


if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        main()
