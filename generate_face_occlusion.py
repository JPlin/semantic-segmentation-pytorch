#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by: jplin
# Microsoft Research & XiaMen University
# jplinforever@gmail.com
# Copyright (c) 2019

import copy
import glob
import os
import pickle
import random
from functools import lru_cache

import tqdm
import cv2
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from skimage import io, transform, img_as_ubyte
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

CELEBA_DIR = "\\\\4CV8116QRZ\\Data\\CelebA_align\\img_align_celeba"
OCCLUSION_DIR = "\\\\msra-facednn03\\v-lingl\\faceswapNext\\data"
OUT_DIR = os.path.join('data', 'celeba_occlu')
HELEN_DIR = os.path.join('data', 'helen_occlu')
CELEBAHQ_DIR = os.path.join('data', 'celeba_hq')
OCCLUSION_WIDTH = 178
OCCLUSION_HEIGHT = 218


class occulusion_augmentor():
    def __init__(self):
        self.all_celeba = []
        for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
            self.all_celeba += glob.glob(os.path.join(CELEBA_DIR,
                                                      '*' + suffix))

        self.all_hand = []
        for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
            for f in ['hand', 'egohands_extracted', 'GTEA_extracted']:
                self.all_hand += glob.glob(
                    os.path.join(OCCLUSION_DIR, f, '*' + suffix))

        self.all_shape = []
        for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
            for f in ['ShapePick']:
                self.all_shape += glob.glob(
                    os.path.join(OCCLUSION_DIR, f, '*' + suffix))

        self.hand_augmentor = iaa.Sequential([
            iaa.PadToFixedSize(OCCLUSION_WIDTH, OCCLUSION_HEIGHT),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-180, 180),
                       scale=(0.4, 0.6),
                       translate_percent={
                           'x': (-0.5, 0.5),
                           'y': (-0.45, 0.45)
                       })
        ])

        self.shape_augmentor = iaa.Sequential([
            iaa.PadToFixedSize(OCCLUSION_WIDTH, OCCLUSION_HEIGHT),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-180, 180),
                       scale=(0.6, 0.8),
                       translate_percent={
                           'x': (-0.45, 0.45),
                           'y': (-0.45, 0.45)
                       })
        ])

    @lru_cache()
    def get_csize(self, height, width):
        u = int(height * 0.47)
        d = int(height * 0.68)
        l = int(width * 0.4)
        r = int(width * 0.6)
        return u, d, l, r

    def blur(self, mask):
        blur_radius = random.choice([1, 3, 5, 7, 9, 11])
        mask = np.array(mask).astype(np.uint8) * 255
        return cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

    def aug_idx(self, idx):
        origin_image = io.imread(self.all_celeba[idx])
        image, mask = self.merge_hand_and_shape(origin_image / 255.,
                                                origin_image / 255.)
        return image, mask

    def merge_hand_and_shape(self, image, origin_image):
        all_alpha = np.zeros((image.shape[0], image.shape[1], 1))

        u, d, l, r = self.get_csize(image.shape[0], image.shape[1])
        hands_num = random.choices(
            population=[0, 1, 2, 3],
            weights=[0.2, 0.7, 0.05, 0.05],
        )[0]

        if hands_num >= 0:
            origin_image_skin_mean = np.mean(origin_image[u:d, l:r, :], (0, 1))
            origin_image_skin_variance = np.std(origin_image[u:d, l:r, :],
                                                (0, 1))
            for _ in range(hands_num):
                hand = io.imread(random.choice(self.all_hand))
                height, width, _ = hand.shape
                long_edge = max(height, width)
                scale = image.shape[1] / long_edge
                hand = transform.rescale(hand, scale=scale)

                t_mean = origin_image_skin_mean
                t_variance = origin_image_skin_variance
                hand_mean = np.mean(hand[:, :, :3][hand[:, :, 3] != 0], 0)
                hand_variance = np.std(hand[:, :, :3][hand[:, :, 3] != 0], 0)

                hand = self.hand_augmentor.augment_image(hand)
                hand, alpha = hand[:, :, :3], hand[:, :, 3:]
                match_hand = (((hand - hand_mean) /
                               (hand_variance + 1e-6)) * t_variance + t_mean)
                blend_alpha = np.expand_dims(self.blur(alpha), -1) / 255
                #print(np.max(blend_alpha))
                image = image * (1 - blend_alpha) + match_hand * blend_alpha
                all_alpha += alpha

        shape_num = random.choices(population=[0, 1, 2, 3],
                                   weights=[0.4, 0.5, 0.05, 0.05])[0]
        if shape_num >= 0:
            origin_image_mean = np.mean(origin_image[u:d, l:r, :], (0, 1))
            origin_image_variance = np.std(origin_image[u:d, l:r, :], (0, 1))
            for _ in range(shape_num):

                this_shape = io.imread(random.choice(self.all_shape))

                height, width, _ = this_shape.shape
                long_edge = max(height, width)
                scale = image.shape[1] / long_edge
                this_shape = transform.rescale(this_shape, scale=scale)

                this_shape_mean = np.mean(
                    this_shape[:, :, :3][this_shape[:, :, 3] != 0], 0)
                this_shape_variance = np.std(
                    this_shape[:, :, :3][this_shape[:, :, 3] != 0], 0)
                r_num = random.random()
                if r_num < 0.5:
                    t_mean = this_shape_mean
                    t_variance = this_shape_variance
                else:
                    t_mean = origin_image_mean
                    t_variance = origin_image_variance

                this_shape = self.shape_augmentor.augment_image(this_shape)
                this_shape, alpha = this_shape[:, :, :3], this_shape[:, :, 3:]
                match_this_shape = (
                    (this_shape - this_shape_mean) /
                    (this_shape_variance + 1e-6)) * t_variance + t_mean
                blend_alpha = np.expand_dims(self.blur(alpha), -1) / 255
                image = image * (1 -
                                 blend_alpha) + match_this_shape * blend_alpha
                all_alpha += alpha
        image = np.clip(image, a_min=0, a_max=1)
        all_alpha[all_alpha > 0.01] = 1
        all_alpha[all_alpha < 0.01] = 0
        return image, all_alpha


def aug_face_occulsion(times=1):
    oa = occulusion_augmentor()
    for time in range(times):
        for i in tqdm.tqdm(range(len(oa.all_celeba))):
            image, mask = oa.aug_idx(i)
            # plt.imshow((image * 255).astype(np.uint8))
            # plt.show()
            # plt.imshow(mask.squeeze())
            # plt.show()
            io.imsave(os.path.join(OUT_DIR, f'{i:05d}_{time}.jpg'),
                      img_as_ubyte(image))
            io.imsave(os.path.join(OUT_DIR, f'{i:05d}_{time}.png'),
                      mask.squeeze().astype(np.uint8))


def append_odgt(txt_path, path_list, new=False):
    import json
    open_mode = 'w+' if new else 'a+'
    with open(txt_path, open_mode) as outfile:
        for image_pth in path_list:
            if os.path.exists(image_pth) and os.path.exists(
                    image_pth.replace('.jpg', '.png')):
                image = io.imread(image_pth)
                H, W = image.shape[:2]
                record = json.dumps({
                    "fpath_img":
                    image_pth,
                    "fpath_segm":
                    image_pth.replace('.jpg', '.png'),
                    "width":
                    W,
                    "height":
                    H
                })
                outfile.write(record + '\n')


def generate_json_file(split_rate=0.99):
    train_txt = os.path.join('data', 'face_train_part.odgt')
    validate_txt = os.path.join('data', 'face_validate_part.odgt')

    all_celeba = glob.glob(os.path.join(OUT_DIR, '*.jpg'))
    # all_image = all_image[:100]
    length = 15000
    append_odgt(train_txt, all_celeba[:int(length * 0.99)], new=True)
    append_odgt(validate_txt, all_celeba[int(length * 0.99):], new=True)

    all_celebahq = glob.glob(os.path.join(CELEBAHQ_DIR, '*.jpg'))
    # all_image = all_image[:100]
    length = len(all_celebahq)
    append_odgt(train_txt, all_celebahq[:int(length * 0.99)])
    append_odgt(validate_txt, all_celebahq[int(length * 0.99):])

    all_helen = glob.glob(os.path.join(HELEN_DIR, '*.jpg'))
    # all_image = all_image[:100]
    length = len(all_helen)
    append_odgt(train_txt, all_helen[:int(length * 0.99)])
    append_odgt(validate_txt, all_helen[int(length * 0.99):])


if __name__ == "__main__":
    # aug_face_occulsion()
    generate_json_file()
