import os
import random
import numpy as np
from PIL import Image

def gen_pair_im(self, lr_path, hr_path):
    lr = Image.open(lr_path)
    # cubic = lr.resize(lr.size * 2, Image.CUBIC)
    cubic = lr.resize((lr.size[0] * 2, lr.size[1] * 2), Image.CUBIC)

    lr = lr.convert('YCbCr')
    lr = np.expand_dims(np.array(lr.split()[0]), -1)
    cubic = cubic.convert('YCbCr')
    cubic = np.expand_dims(np.array(cubic.split()[0]), -1)

    hr = Image.open(hr_path).convert('YCbCr')
    hr = np.expand_dims(np.array(hr.split()[0]), -1)

    hr, lr, cubic = random_crop(hr, lr, cubic, size=self.crop_size, scale=self.scale)
    hr, lr, cubic = random_flip_and_rotate(hr, lr, cubic)

    hr = hr / 127.5 - 1.
    lr = lr / 127.5 - 1.
    cubic = cubic / 127.5 - 1.
    return hr, lr, cubic

def gen_pair_im_1(self, lr, hr, size=128, scale=2):
    cubic = tf.image.resize_bicubic(lr, (size * 2, size * 2))

    lr = tf.image.rgb_2_yuv(lr)
    lr = np.expand_dims(np.array(lr.split()[0]), -1)
    cubic = cubic.convert('YCbCr')
    cubic = np.expand_dims(np.array(cubic.split()[0]), -1)

    hr = lr.convert('YCbCr')
    hr = np.expand_dims(np.array(hr.split()[0]), -1)

    hr, lr, cubic = random_crop(hr, lr, cubic, size=self.crop_size, scale=self.scale)
    hr, lr, cubic = random_flip_and_rotate(hr, lr, cubic)

    hr = hr / 127.5 - 1.
    lr = lr / 127.5 - 1.
    cubic = cubic / 127.5 - 1.
    return hr, lr, cubic

def random_crop(hr, lr, cubic, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    crop_lr = lr[y: y + size, x: x + size].copy()
    crop_cubic = cubic[hy: hy + hsize, hx: hx + hsize].copy()
    crop_hr = hr[hy: hy + hsize, hx: hx + hsize].copy()

    return crop_hr, crop_lr, crop_cubic


def random_flip_and_rotate(hr, lr, cubic):
    if random.random() < 0.5:
        hr = np.flipud(hr)
        lr = np.flipud(lr)
        cubic = np.flipud(cubic)

    if random.random() < 0.5:
        hr = np.fliplr(hr)
        lr = np.fliplr(lr)
        cubic = np.fliplr(cubic)

    angle = random.choice([0, 1, 2, 3])
    hr = np.rot90(hr, angle)
    lr = np.rot90(lr, angle)
    cubic = np.rot90(cubic, angle)

    return hr.copy(), lr.copy(), cubic.copy()

def tf_random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def tf_random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def tf_random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)
