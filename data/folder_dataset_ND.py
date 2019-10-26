#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by: algohunt
# Microsoft Research & Peking University 
# lilingzhi@pku.edu.cn
# Copyright (c) 2019

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import zipfile
import os
from PIL import Image
import glob
import pickle
import torch
from config import config as cfg
import copy
import torch
import imgaug.augmenters as iaa
from skimage import transform,io
from PIL import Image
import numpy as np
import cv2
from functools import lru_cache

class Folder_Dataset(Dataset):

    def __init__(self,base_folder,transform=None,return_name=False,
           folder_names=None,aug_hand=False,use_valid_key=True):
        super().__init__()
        self.paths = []
        if folder_names is None:
            folder_names = cfg.sub_folders
        all_folder = folder_names

        if isinstance(folder_names[0], dict):
            all_folder = list(next(iter(d.keys())) for d in folder_names)
            print(all_folder)
        pkl_name = ('paths_of_'+'_'.join(all_folder) +'_'+ '.pkl').replace('/', '_')

        pkl_file = os.path.join(base_folder,pkl_name)


        if not os.path.exists(pkl_file):
            for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
                self.paths += glob.glob(os.path.join(base_folder, '*' + suffix), recursive=True)

            for sub_folder in all_folder:
                for suffix in ['.jpg','.png','.JPG','.PNG']:
                    self.paths += glob.glob(os.path.join(base_folder,sub_folder,'**','*'+suffix),recursive=True)
            self.paths = list(set(self.paths))
            self.paths.sort()
            with open(pkl_file, "wb") as fp:  # Unpickling
                pickle.dump(self.paths, fp)

        with open(pkl_file, "rb") as fp:  # Unpickling
            self.paths = pickle.load(fp)
        print('load from existing {}'.format(pkl_file))

        print('dataset size is ',len(self.paths))
        if isinstance(folder_names[0], dict):
            new_paths = copy.deepcopy(self.paths)
            all_k = {}
            for d in folder_names:
                for k,v in d.items():
                    if v>1:
                        all_k[k] = v
            for p in self.paths :
                for k,v in all_k.items():
                    if k in p:
                        new_paths.extend([p]*(v-1))
            self.paths = new_paths
            print('after re ordering dataset\'s size is ', len(self.paths))

        self.paths.sort()
        self.transform = transform
        self.return_name=return_name



        self.bias = 0


        self.occ_aug = occulusion_augmentor()



    def __getitem__(self, item):
        path = self.paths[item+self.bias]


        if not os.path.exists(path):
            return self[item+1]


        image = io.imread(path)/255
        origin_image = copy.deepcopy(image)
        image = transform.resize(image, (cfg.imsize, cfg.imsize))
        if random.random() > (1 - cfg.augmentation.hand.rate):

            image, all_alpha = self.occ_aug.merge_hand_and_shape(image=image, origin_image=origin_image)

        else:
            all_alpha = np.zeros((image.shape[0],image.shape[1], 1))
            image = copy.deepcopy(origin_image)

        img = Image.fromarray((image * 255).astype(np.uint8))
        origin_image = Image.fromarray((origin_image * 255).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
            origin_image = self.transform(origin_image)
        all_alpha = torch.from_numpy(all_alpha)

        all_alpha = all_alpha.permute(2,0,1).float()
        name = os.path.basename(path)
        return tuple([img, name, all_alpha, origin_image])


    def get_img_by_key(self, key):
        if not self.keys:
            print("generating keys")
            self.keys = {os.path.basename(p).split('.')[0]:idx for idx,p in enumerate(self.paths)}
        idx = self.keys[key]
        idx -= self.bias
        return self.__getitem__(idx)


    def __len__(self):
        return len(self.paths)

@lru_cache()
def get_csize(height,width):
    u = int(height * 0.47)
    d = int(height * 0.68)
    l = int(width * 0.4)
    r = int(width * 0.6)
    return u, d, l, r


#@lru_cache(maxsize=10000)
def cache_skread(p):
    return io.imread(p)/255

import random

class Folder_Dataset_Pair(Dataset):
    def __init__(self,folder,transform=None,return_name=False, aug_hand=False):
        super().__init__()
        self.paths = []

        pkl_file = os.path.join(folder,'paths_sp.pkl')
        if os.path.exists(pkl_file) :
            with open(pkl_file, "rb") as fp:  # Unpickling
                self.paths = pickle.load(fp)
            print('load from existing paths_sp.pkl')
        else:
            for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
                self.paths += glob.glob(os.path.join(folder, '**', 'vggface2_big_remove_padding', '*' + suffix),
                                        recursive=True)
            self.paths.sort()
            with open(pkl_file, "wb") as fp:  # Unpickling
                pickle.dump(self.paths,fp)

        self.p_dict = {}
        for p in self.paths:
            k = os.path.basename(p)[:7]
            if k in self.p_dict:
                self.p_dict[k].append(p)
            else:
                self.p_dict[k] = [p]
        self.id_list = list(self.p_dict.keys())
        self.transform = transform
        self.return_name = return_name
        self.valid_key = set()
        pkl_file = os.path.join(cfg.path.data_dir, cfg.id_encoder_type + '_id_emb.pkl')
        if os.path.exists(pkl_file):
            all_id_emb = torch.load(pkl_file)
            self.valid_key = set(all_id_emb.keys())
            print('valid key size: ', len(self.valid_key))
        self.occ_aug = occulusion_augmentor()

    def __getitem__(self, item):

        this_id = self.id_list[item]
        while True:

            p1 = random.choice(self.p_dict[this_id])
            key_name = os.path.basename(p1).split('.')[0]
            if key_name in self.valid_key:
                break
            this_id = random.choice(self.id_list)
        while True:
            p2 = random.choice(self.p_dict[this_id])
            key_name = os.path.basename(p1).split('.')[0]
            if key_name in self.valid_key:
                break

        img1 = Image.open(p1).convert('RGB')



        img2 = io.imread(p2)/255
        origin_img2 = copy.deepcopy(img2)
        img2 = transform.resize(img2, (cfg.imsize, cfg.imsize))
        if random.random() > (1 - cfg.augmentation.hand.rate):
            img2,all_alpha = self.occ_aug.merge_hand_and_shape(image=img2, origin_image=origin_img2)
        else:
            all_alpha = np.zeros((img2.shape[0],img2.shape[1], 1))
        img2 = Image.fromarray((img2 * 255).astype(np.uint8))
        all_alpha = torch.from_numpy(all_alpha).permute(2,0,1).float()

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return tuple([img1, img2, os.path.basename(p1), os.path.basename(p2), all_alpha])


    def __len__(self):
        return len(self.id_list)

class occulusion_augmentor():

    def __init__(self):
        self.all_hand = []
        for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
            for f in ['hand', 'egohands_extracted', 'GTEA_extracted']:
                self.all_hand += glob.glob(os.path.join(cfg.path.data_dir, f, '*' + suffix))

        self.all_shape = []
        for suffix in ['.jpg', '.png', '.JPG', '.PNG']:
            for f in ['ShapePick']:
                self.all_shape += glob.glob(os.path.join(cfg.path.data_dir, f, '*' + suffix))

        self.hand_augmentor = iaa.Sequential(
            [
                iaa.PadToFixedSize(cfg.imsize,cfg.imsize),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), scale=(0.4, 0.6),
                           translate_percent={'x': (-0.5, 0.5), 'y': (-0.5, 0.5)})
            ]
        )

        self.shape_augmentor = iaa.Sequential(
            [
                iaa.PadToFixedSize(cfg.imsize, cfg.imsize),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), scale=(0.6, 0.8),
                           translate_percent={'x': (-0.5, 0.5), 'y': (-0.5, 0.5)})
            ]
        )

    def merge_hand_and_shape(self,image, origin_image):
        all_alpha = np.zeros((image.shape[0], image.shape[1],1))

        u, d, l, r = get_csize(image.shape[0],image.shape[1])
        hands_num = random.choices(population=[0, 1, 2, 3],
                                   weights=cfg.augmentation.hand.chosse_weight,
                                   )[0]

        if hands_num >= 0:
            origin_image_skin_mean = np.mean(origin_image[u:d, l:r, :], (0, 1))
            origin_image_skin_variance = np.std(origin_image[u:d, l:r, :], (0, 1))
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
                hand, alpha = hand[:, :, :3], hand[:, :, 3:]
                match_hand = (((hand - hand_mean) / (hand_variance + 1e-6)) * t_variance + t_mean)
                blend_alpha = np.expand_dims(blur(alpha),-1)/255
                #print(np.max(blend_alpha))
                image = image * (1 - blend_alpha) + match_hand * blend_alpha
                all_alpha += alpha

        shape_num = random.choices(population=[0, 1, 2, 3],
                                   weights=cfg.augmentation.hand.chosse_weight
                                   )[0]
        if shape_num >= 0:
            origin_image_mean = np.mean(origin_image[u:d, l:r, :], (0, 1))
            origin_image_variance = np.std(origin_image[u:d, l:r, :], (0, 1))
            for _ in range(shape_num):

                this_shape = io.imread(random.choice(self.all_shape))

                height, width, _ = this_shape.shape
                long_edge = max(height, width)
                scale = image.shape[0] / long_edge
                this_shape = transform.rescale(this_shape, scale=scale)

                this_shape_mean = np.mean(this_shape[:, :, :3][this_shape[:, :, 3] != 0], 0)
                this_shape_variance = np.std(this_shape[:, :, :3][this_shape[:, :, 3] != 0], 0)
                r_num = random.random()
                if r_num < 0.5 :
                    t_mean = this_shape_mean
                    t_variance = this_shape_variance
                else :
                    t_mean = origin_image_mean
                    t_variance = origin_image_variance


                this_shape = self.shape_augmentor.augment_image(this_shape)
                this_shape, alpha = this_shape[:, :, :3], this_shape[:, :, 3:]
                match_this_shape = ((this_shape - this_shape_mean) / (this_shape_variance + 1e-6)) * t_variance + t_mean
                blend_alpha = np.expand_dims(blur(alpha),-1)/255
                image = image * (1 - blend_alpha) + match_this_shape * blend_alpha
                all_alpha += alpha
        image = np.clip(image, a_min=0, a_max=1)
        all_alpha[all_alpha > 0.01] = 1
        all_alpha[all_alpha < 0.01] = 0
        return image, all_alpha


def blur(mask):
    blur_radius = random.choice([1,3,5,7,9, 11])
    mask = np.array(mask).astype(np.uint8)*255
    return cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)