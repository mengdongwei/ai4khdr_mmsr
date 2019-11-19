config_edsr = {
    'in_channels': 3,
    'base_channels': 64,
    'n_resblocks': 8,

    'batch': 64,
    'lr': 0.0002,
    'loss_fn': 'L1',
    'lr_epochs': [100, 200, 25],
    'max_epochs': 300,
    'gamma': 0.5,

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}

config_msrn = {
    'in_channels': 3,
    'base_channels': 64,
    'n_resblocks': 8,

    'batch': 64,
    'lr': 0.0002,
    'loss_fn': 'L1',
    'lr_epochs': [100, 200, 250],
    'max_epochs': 300,
    'gamma': 0.5,

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}

config_wdsr = {
    'in_channels': 1,
    'base_channels': 32,
    'res_block_expansion': 6,
    'n_resblocks': 8,
    'res_block_type': 'wdsr_b',  # 'wdsr_a', expansion=4

    'batch': 32,
    'lr': 0.0001,
    'num_iter': 25000,
    'lr_decay_step': 40000,
    'loss_fn': 'L1',

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
    'weight_initialize_scale': 0.1,
}

config_wdsrd = {
    'in_channels': 1,
    'base_channels': 32,
    'res_block_expansion': 6,
    'n_resblocks': 8,
    'res_block_type': 'wdsr_b',  # 'wdsr_a', expansion=4

    'batch': 32,
    'lr': 0.0004,
    'num_iter': 25000,
    'lr_decay_step': 10000,
    'loss_fn': 'L1',

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}

config_esrgan = {
    'in_channels': 1,
    'n_filter': 64,
    'inc_filter': 32,
    'num_repeat_RRDB': 1,
    'residual_scaling': 0.2,
    'weight_initialize_scale': 0.1,

    'batch': 32,
    'lr': 0.0002,
    'num_iter': 50000,
    'lr_decay_step': 20000,
    'loss_fn': 'L1',

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}

config_small = {
    'in_channels': 1,
    'channels_enc': 16,
    'channels_dec': 8,

    'res_block_expansion': 6,
    'n_resblocks': 4,

    'batch': 32,
    'lr': 0.0004,
    'num_iter': 25000,
    'lr_decay_step': 10000,
    'loss_fn': 'L1',

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
    'weight_initialize_scale': 0.1,
}

config_wdsr_gan = {
    'in_channels': 1,
    'base_channels': 32,
    'res_block_expansion': 6,
    'n_resblocks': 8,
    'res_block_type': 'wdsr_b',  # 'wdsr_a', expansion=4
    'weight_initialize_scale': 0.1,

    'batch': 32,
    'lr': 0.0001,
    'num_iter': 25000,
    'lr_decay_step': 10000,
    'loss_fn': 'L1',

    'gan_loss_type': 'RaGAN',
    'perceptual_loss' : 'VGG19',
    'pre_train_checkpoint_dir': '',

    'epsilon': 1e-12,
    'gan_loss_coeff': 0.005,
    'content_loss_coeff': 0.01,

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}

config_wdsrd_gan = {
    'in_channels': 1,
    'base_channels': 32,
    'res_block_expansion': 6,
    'n_resblocks': 8,
    'res_block_type': 'wdsr_b',  # 'wdsr_a', expansion=4
    'weight_initialize_scale': 0.1,

    'batch': 32,
    'lr': 0.0001,
    'num_iter': 25000,
    'lr_decay_step': 80000,
    'loss_fn': 'L1',

    'gan_loss_type': 'RaGAN',
    'perceptual_loss' : 'VGG19',
    'pre_train_checkpoint_dir': '',

    'epsilon': 1e-12,
    'gan_loss_coeff': 0.005,
    'content_loss_coeff': 0.01,

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}

config_esrgan_gan = {
    'in_channels': 1,
    'n_filter': 64,
    'inc_filter': 32,
    'num_repeat_RRDB': 1,
    'residual_scaling': 0.2,
    'weight_initialize_scale': 0.1,

    'batch': 16,
    'lr': 0.0001,
    'num_iter': 50000,
    'lr_decay_step': 20000,
    'loss_fn': 'L1',

    'gan_loss_type': 'RaGAN',
    'perceptual_loss' : 'VGG19',
    'pre_train_checkpoint_dir': '',

    'epsilon': 1e-12,
    'gan_loss_coeff': 0.005,
    'content_loss_coeff': 0.01,

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}

config_small_gan = {
    'in_channels': 1,
    'channels_enc': 16,
    'channels_dec': 8,

    'res_block_expansion': 6,
    'n_resblocks': 4,

    'batch': 32,
    'lr': 0.0001,
    'num_iter': 25000,
    'lr_decay_step': 10000,
    'loss_fn': 'L1',

    'gan_loss_type': 'RaGAN',
    'perceptual_loss': 'VGG19',
    'pre_train_checkpoint_dir': '',

    'epsilon': 1e-12,
    'gan_loss_coeff': 0.005,
    'content_loss_coeff': 0.01,

    'print_interval': 50,
    'eval_interval': 1,
    'save_interval': 1,
}
