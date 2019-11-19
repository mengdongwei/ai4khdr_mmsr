from data.LQGT_dataset import LQGTDataset
from utils import crash_on_ipy

import options.options as option

opt_file = './options/train/train_ESRGAN_M.yml'
opt = option.parse(opt_file, is_train=True)
test_set = LQGTDataset(opt['datasets']['train'])

for xx in test_set:
    print(type(xx), len(xx), xx['LQ'].shape, xx['GT'].shape)
