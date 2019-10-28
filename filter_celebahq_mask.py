import glob
import os
from tqdm import tqdm
from os.path import join as opj
from pathlib import Path

import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import numpy as np
from skimage import io, transform

CELEBA_DIR = "\\\\4CV8116QRZ\\Data\\CelebAMask-HQ\\CelebAMask-HQ"
OUT_DIR = "data/celeba_hq"

seq = iaa.Sequential([
    iaa.Affine(rotate=(-20, 20),
               scale=(0.8, 1.1),
               translate_percent={
                   'x': (-0.1, 0.1),
                   'y': (-0.1, 0.1)
               }),
    iaa.OneOf([iaa.GaussianBlur((0, 1.0))])
])


def extract_eyegalss(expansion=3):
    eye_gs = glob.glob(
        opj(CELEBA_DIR, 'CelebAMask-HQ-mask-anno', '*/*eye_g.png'))
    print('total length:', len(eye_gs))
    for mask_pth in tqdm(eye_gs):
        idx = int(os.path.basename(mask_pth).split('_')[0])
        mask = io.imread(mask_pth)
        image = io.imread(opj(CELEBA_DIR, 'CelebA-HQ-img', str(idx) + '.jpg'))
        image = transform.rescale(image, 0.5)
        mask = transform.rescale(mask, 0.5, order=0)[:, :, 0].astype(np.uint8)

        images = np.expand_dims(image, 0)
        masks = np.expand_dims(np.expand_dims(mask, 0), -1)
        for i in range(expansion):
            image, mask = seq(images=images, segmentation_maps=masks)
            # plt.imshow(image.squeeze())
            # plt.show()
            # plt.imshow(mask.squeeze().astype(np.uint8))
            # plt.show()
            io.imsave(opj(OUT_DIR, str(idx) + f'_{i}.jpg'), image.squeeze())
            io.imsave(opj(OUT_DIR,
                          str(idx) + f'_{i}.png'),
                      mask.squeeze().astype(np.uint8))


if __name__ == '__main__':
    extract_eyegalss()
