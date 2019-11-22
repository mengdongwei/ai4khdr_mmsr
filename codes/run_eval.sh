# single GPU training (image SR)
#python train.py -opt options/train/train_SRResNet.yml
#python train.py -opt options/train/train_SRGAN.yml
#python train.py -opt options/train/train_ESRGAN.yml


# distributed training (video SR)
# 8 GPUs
python -m torch.distributed.launch --nproc_per_node=1 --master_port=66666 eval_demo/test_ai4khdr_valid.py --launcher pytorch
