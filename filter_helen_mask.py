import glob
import os
import pickle
import random
from functools import lru_cache
from os.path import join as opj
from pathlib import Path
import sys
from tqdm import tqdm

import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import img_as_ubyte, io, transform
sys.path.append('E:\\v-jinpl\\libcv')
from facealignment import FA

HELEN_DIR = "\\\\MSRA-FACEDNN05/haya2/Datasets/SmithCVPR2013_dataset_resized/images_no_occ"
ALIGN_DIR = "data/helen_occlu_align"
OCCLUSION_DIR = "\\\\msra-facednn03\\v-lingl\\faceswapNext\\data"
OUT_DIR = "data/helen_occlu"
OCCLUSION_WIDTH = 256
OCCLUSION_HEIGHT = 256


class occulusion_augmentor():
    def __init__(self):
        self.all_helen = []
        for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
            self.all_helen += glob.glob(os.path.join(ALIGN_DIR, '*' + suffix))

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
                       scale=(0.3, 0.5),
                       translate_percent={
                           'x': (-0.4, 0.4),
                           'y': (-0.4, 0.4)
                       })
        ])

        self.shape_augmentor = iaa.Sequential([
            iaa.PadToFixedSize(OCCLUSION_WIDTH, OCCLUSION_HEIGHT),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-180, 180),
                       scale=(0.3, 0.6),
                       translate_percent={
                           'x': (-0.4, 0.4),
                           'y': (-0.45, 0.45)
                       })
        ])

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        self.image_augmentor = iaa.Sequential([
            iaa.Affine(rotate=(-18, 18),
                       scale=(0.8, 1.2),
                       translate_percent={
                           'x': (-0.08, 0.08),
                           'y': (-0.08, 0.08)
                       }),
            iaa.SomeOf((0, 1), [
                iaa.Invert(0.05, per_channel=True),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(1, 3)), 
                    iaa.MedianBlur(k=(1, 5)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
            ])
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
        origin_image = io.imread(self.all_helen[idx])
        # origin_image = self.image_augmentor.augment_image(origin_image)
        image, mask = self.merge_hand_and_shape(origin_image / 255.,
                                                origin_image / 255.)
        image = (image * 255).astype(np.uint8)
        mask = mask.astype(np.uint8)

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        image, mask = self.image_augmentor(images=image, segmentation_maps=mask)
        return image.squeeze(0), mask.squeeze()

    def merge_hand_and_shape(self, image, origin_image):
        all_alpha = np.zeros((image.shape[0], image.shape[1], 1))

        u, d, l, r = self.get_csize(image.shape[0], image.shape[1])
        hands_num = random.choices(
            population=[0, 1, 2, 3],
            weights=[0.1, 0.8, 0.05, 0.05],
        )[0]

        if hands_num >= 0:
            origin_image_skin_mean = np.mean(origin_image[u:d, l:r, :], (0, 1))
            origin_image_skin_variance = np.std(origin_image[u:d, l:r, :],
                                                (0, 1))
            for _ in range(hands_num):
                hand = io.imread(random.choice(self.all_hand))
                height, width, _ = hand.shape
                long_edge = max(height, width)
                scale = image.shape[0] / long_edge
                hand = transform.rescale(hand, scale=scale)

                t_mean = origin_image_skin_mean
                t_variance = origin_image_skin_variance
                hand_mean = np.mean(hand[:, :, :3][hand[:, :, 3] != 0], 0)
                hand_variance = np.std(hand[:, :, :3][hand[:, :, 3] != 0], 0)

                hand = self.hand_augmentor.augment_image(hand)
                hand = iaa.PadToFixedSize(image.shape[1],
                                          image.shape[0]).augment_image(hand)
                hand = iaa.CropToFixedSize(image.shape[1],
                                           image.shape[0]).augment_image(hand)
                hand, alpha = hand[:, :, :3], hand[:, :, 3:]
                match_hand = (((hand - hand_mean) /
                               (hand_variance + 1e-6)) * t_variance + t_mean)
                blend_alpha = np.expand_dims(self.blur(alpha), -1) / 255
                #print(np.max(blend_alpha))
                image = image * (1 - blend_alpha) + match_hand * blend_alpha
                all_alpha += alpha

        shape_num = random.choices(population=[0, 1, 2, 3],
                                   weights=[0.4, 0.3, 0.05, 0.05])[0]
        if shape_num >= 0:
            origin_image_mean = np.mean(origin_image[u:d, l:r, :], (0, 1))
            origin_image_variance = np.std(origin_image[u:d, l:r, :], (0, 1))
            for _ in range(shape_num):

                this_shape = io.imread(random.choice(self.all_shape))

                height, width, _ = this_shape.shape
                long_edge = max(height, width)
                scale = image.shape[0] / long_edge
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
                this_shape = iaa.PadToFixedSize(
                    image.shape[1], image.shape[0]).augment_image(this_shape)
                this_shape = iaa.CropToFixedSize(
                    image.shape[1], image.shape[0]).augment_image(this_shape)
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


def aug_face_occulsion(times=6):
    oa = occulusion_augmentor()
    for time in range(times):
        for i in tqdm(range(len(oa.all_helen))):
            image, mask = oa.aug_idx(i)
            # plt.imshow(image)
            # plt.show()
            # plt.imshow(mask.squeeze())
            # plt.show()
            io.imsave(os.path.join(OUT_DIR, f'{i:05d}_{time}.jpg'),
                      image)
            io.imsave(os.path.join(OUT_DIR, f'{i:05d}_{time}.png'),
                      mask.squeeze().astype(np.uint8))


def _meshgrid(h, w):
    yy, xx = np.meshgrid(np.arange(0, h, dtype=np.float32),
                         np.arange(0, w, dtype=np.float32),
                         indexing='ij')
    return yy, xx


def _warp_meshgrid(h, w, transform_matrix):
    yy, xx = _meshgrid(h, w)
    yy = np.reshape(yy, [-1])
    xx = np.reshape(xx, [-1])
    xxyy_one = np.stack([xx, yy, np.ones_like(xx)], axis=0)  # 3x(h*w)

    inv_matrix = np.linalg.inv(transform_matrix)
    xxyy_one = np.dot(inv_matrix, xxyy_one)
    xx = np.reshape(
        np.divide(xxyy_one[0, :], xxyy_one[2, :], dtype=np.float32), [h, w])
    yy = np.reshape(
        np.divide(xxyy_one[1, :], xxyy_one[2, :], dtype=np.float32), [h, w])
    return yy, xx


def align_helen():
    fa = FA(gpu=True)
    all_helen = []
    for suffix in ['.jpg']:
        all_helen += glob.glob(os.path.join(HELEN_DIR, '*' + suffix))

    for helen in tqdm(all_helen):
        image = io.imread(helen)
        base_name = os.path.basename(helen)[:-len('.jpg')]
        five_points = FA.get_five_points(fa.get_landmark(image))
        out_img, _ = FA.extract_image_chip(image, five_points, padding=-0.07)

        # plt.imshow(out_img)
        # plt.show()
        io.imsave(os.path.join(ALIGN_DIR, base_name + '.jpg'), out_img)


if __name__ == '__main__':
    aug_face_occulsion()
    # align_helen()
