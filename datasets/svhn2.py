###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Classes and functions used to create the Street View House Numbers (SVHN) Dataset.
(http://ufldl.stanford.edu/housenumbers/)
Format: 1 is used: Format with Bounding Boxes
"""
import os

import h5py
import numpy as np
import pandas as pd
import sys

import albumentations as album
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import ai8x

class SVHN(Dataset):
    """
    Street View House Numbers (SVHN) Dataset. (http://ufldl.stanford.edu/housenumbers/)
    Format: 1 is used: Format with Bounding Boxes

    Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits
    in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and
    Unsupervised Feature Learning 2011.
    """

    bad_img_list = ['8066.png', '32485.png']

    def __init__(self, root_dir, d_type, transform=None, resize_size=(96, 96), augment_img=False):
        
        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        self.box_size_to_crop_img_ratio = 2
        
        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.resize_size = resize_size

        self.img_list = []
        self.boxes_list = []
        self.lbls_list = []

        self.data_dir = os.path.join(self.root_dir, self.__class__.__name__, self.d_type)
        self.mat_file_path = os.path.join(self.data_dir, 'digitStruct.mat')
        self.gt_file_path = os.path.join(self.root_dir, self.__class__.__name__, 'processed', self.d_type+'_data_gt.csv') 
        
        if not self.__check_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!")

        self.__prepare_ground_truths()

        self.c_transforms = None
        if augment_img:
            self.c_transforms = \
                album.Compose([album.RandomBrightnessContrast(p=0.5),
                               album.RGBShift(r_shift_limit=64, g_shift_limit=64,
                                              b_shift_limit=64, p=0.7),
                               album.ColorJitter(brightness=0.5, contrast=0.5,
                                                 saturation=0.5, hue=0.5, p=0.7),
                               album.CLAHE(p=0.7),
                               album.MultiplicativeNoise(multiplier=(0.25, 0.75), per_channel=True,
                                                         elementwise=True, p=0.7),
                               album.MotionBlur(p=0.7),
                               album.AdvancedBlur(p=0.9),
                               album.InvertImg(p=0.5)])

    def __check_data_exist(self):
        if os.path.isdir(self.data_dir):
            if os.path.exists(self.mat_file_path):
                return True
        
        return False

    def __print_download_manual(self):
        print('\nDownload the archive file from: '
              'http://ufldl.stanford.edu/housenumbers/[train or test].tar.gz\n'
              'Review the terms and conditions on '
              'http://ufldl.stanford.edu/housenumbers/ and then download...\n'
              'Extract the downloaded archive to path /data/SVHN\n'
              'E.g. The training image files and digitStruct.mat file containing all '
              'annotations will reside under folder: /data/SVHN/training\n')

    def __prepare_ground_truths(self):
        if os.path.exists(self.gt_file_path):
            self.gt = pd.read_csv(self.gt_file_path)
            self.__cvt_gt_string_to_int()
            return
        
        self.gt = SVHN.read_digit_mat(self.mat_file_path)

        # Eleminate the rows with no image name
        self.gt = self.gt.dropna(subset=["img_name"]).reset_index(drop=True)
        # Eliminate some entries as some entries (very few but has to be eliminated) have -1
        neg_indexes = self.gt[self.gt['x0'].apply(lambda list: any(n < 0 for n in list))].index

        # Delete these row indexes from dataFrame
        self.gt.drop(neg_indexes, inplace=True)
        self.gt.reset_index(drop=True, inplace=True)

        self.gt['num_of_boxes'] = self.gt['label'].apply(len)

        # self.gt['img_width'], self.gt['img_height'] = \
        #     zip(*self.gt['img_name'].apply(lambda name: SVHN.get_image_size(
        #         os.path.join(self.root_dir, self.__class__.__name__, self.d_type, name))))

        self.gt['bb_x0'] = self.gt['x0'].apply(min).apply(int)
        self.gt['bb_y0'] = self.gt['y0'].apply(min).apply(int)
        self.gt['bb_x1'] = self.gt['x1'].apply(max).apply(int)
        self.gt['bb_y1'] = self.gt['y1'].apply(max).apply(int)

        self.gt.to_csv(self.gt_file_path, index=False)

    def __cvt_gt_string_to_int(self):
        def str2int(x):
            return np.array(x.strip('[]').split()).astype(np.int32)
        
        self.gt.label = self.gt.label.apply(str2int)
        self.gt.width = self.gt.width.apply(str2int)
        self.gt.height = self.gt.height.apply(str2int)
        self.gt.x0 = self.gt.x0.apply(str2int)
        self.gt.y0 = self.gt.y0.apply(str2int)
        self.gt.x1 = self.gt.x1.apply(str2int)
        self.gt.y1 = self.gt.y1.apply(str2int)

    def __crop_resize_img(self, img, gt_row):
        aspect_ratio = self.resize_size[1] / self.resize_size[0]
        
        bb = [gt_row.bb_x0, gt_row.bb_y0, gt_row.bb_x1, gt_row.bb_y1]
        box_width = int(self.box_size_to_crop_img_ratio * (bb[2] - bb[0]))
        box_height = int(self.box_size_to_crop_img_ratio * (bb[3] - bb[1]))

        if (box_height / aspect_ratio) >= box_width:
            expected_height = box_height
            max_y_shift_percent = 25
            
            expected_width = int(box_height / aspect_ratio)
            x_min = bb[2] - expected_width
            x_max = bb[0]
        
            x0 = np.random.randint(x_min, x_max)
            x1 = x0 + expected_width
            y0 = int(bb[1] + box_height * max_y_shift_percent / 100.0 * (2.0 * np.random.rand() - 1.0))
            y1 = y0 + expected_height
        
            img2 = np.zeros((box_height, expected_width, 3), np.uint8)
        
            x_start1 = max(x0, 0)
            x_end1 = min(x1, img.shape[1])
            y_start1 = max(0, y0)
            y_end1 = min(y1, img.shape[0])
        
            x_start2 = max(0, -x0)
            x_end2 = x_start2 + x_end1 - x_start1
            y_start2 = max(0, -y0)
            y_end2 = y_start2 + y_end1 - y_start1
        else:
            expected_width = box_width
            max_x_shift_percent = 25
            
            expected_height = int(box_width * aspect_ratio)
            y_min = bb[3] - expected_height
            y_max = bb[1]
        
            y0 = np.random.randint(y_min, y_max)
            y1 = y0 + expected_height
            x0 = int(bb[0] + box_width * max_x_shift_percent / 100.0 * (2.0 * np.random.rand() - 1.0))
            x1 = x0 + expected_width
        
            img2 = np.zeros((expected_height, box_width, 3), np.uint8)
        
            x_start1 = max(x0, 0)
            x_end1 = min(x1, img.shape[1])
            y_start1 = max(0, y0)
            y_end1 = min(y1, img.shape[0])
        
            x_start2 = max(0, -x0)
            x_end2 = x_start2 + x_end1 - x_start1
            y_start2 = max(0, -y0)
            y_end2 = y_start2 + y_end1 - y_start1

        img2[y_start2:y_end2, x_start2:x_end2, :] = img[y_start1:y_end1, x_start1:x_end1, :]
        bb2 = [x_start2 - x_start1 + bb[0], y_start2 - y_start1 + bb[1], x_start2 - x_start1 + bb[2], y_start2 - y_start1 + bb[3]]

        resize_ratio = img2.shape[0] / self.resize_size[1]
        bb3 = [int(x / resize_ratio) for x in bb2]
        img3 = cv2.resize(img2, self.resize_size)

        num_digits = len(gt_row.label)
        labels = []
        boxes = []
        for i in range(num_digits):
            x0 = (x_start2 - x_start1 + gt_row.x0[i]) / resize_ratio
            x1 = (x_start2 - x_start1 + gt_row.x1[i]) / resize_ratio
            y0 = (y_start2 - y_start1 + gt_row.y0[i]) / resize_ratio
            y1 = (y_start2 - y_start1 + gt_row.y1[i]) / resize_ratio

            # modify boxes wrt cropped region
            x0 = max(0, x0)
            x1 = min(x1, img3.shape[1]-1)
            y0 = max(0, y0)
            y1 = min(y1, img3.shape[0]-1)

            # check if the digit in the cropped image part
            if (x0 < 0 and x1 < 0) or (x0 >= img3.shape[1] and x1 >= img3.shape[1]) or \
               (y0 < 0 and y1 < 0) or (y0 >= img3.shape[0] and y1 >= img3.shape[0]) or \
               (x0 >= x1) or (y0 >= y1):
                continue
            
            labels.append(gt_row.label[i])
            # Normalize boxes
            boxes.append([x0 / img3.shape[1], y0 / img3.shape[0],
                          x1 / img3.shape[1], y1 / img3.shape[0]])
            
        return img3, boxes, labels

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        img_name = self.gt.iloc[index].img_name
        img_path = os.path.join(self.data_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_row = self.gt.iloc[index]

        img, boxes, labels = self.__crop_resize_img(img, gt_row)

        if self.c_transforms:
            transform_out = self.c_transforms(image=img)
            img = transform_out['image']

        if self.transform is not None:
            img = self.transform(img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return img, (boxes, labels)

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function
        (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
        """
        images = []
        boxes_and_labels = []

        for b in batch:
            images.append(b[0])
            boxes_and_labels.append(b[1])

        images = torch.stack(images, dim=0)
        return images, boxes_and_labels
    
    @staticmethod
    def get_name(index, hdf5_data):
        """Retrieve name field from hdf5 data"""
        name = hdf5_data['/digitStruct/name']
        return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])

    @staticmethod
    def get_bbox(index, hdf5_data):
        """Retrieve bounding box field from hdf5 data"""
        bbox_file_keys = ['label', 'left', 'top', 'width', 'height']
        bbox_dict_keys = ['label', 'x0', 'y0', 'width', 'height']
        
        bbox = {}
        item = hdf5_data['digitStruct/bbox'][index].item()
        
        for file_key, dict_key in zip(bbox_file_keys, bbox_dict_keys):
            attr = hdf5_data[item][file_key]
            values = np.array([int(hdf5_data[attr[()][i].item()][()][0][0])
                      for i in range(len(attr))] if len(attr) > 1 else [int(attr[()][0][0])])
            bbox[dict_key] = values

        return bbox

    @staticmethod
    def read_digit_mat(mat_file):
        """Reading digit information from a .mat file"""
        f = h5py.File(mat_file, 'r')
        bbox = '/digitStruct/bbox'
        
        info_df = pd.DataFrame(index=np.arange(f[bbox].shape[0]), columns=['img_name', 'label', 'width', 'height', 'x0', 'y0'])

        for j in range(f[bbox].shape[0]):  # type: ignore # pylint: disable=no-member
            img_name = SVHN.get_name(j, f)
            if img_name in SVHN.bad_img_list:
                print(info_df.iloc[j])
                continue
            bbox = SVHN.get_bbox(j, f)

            bbox['img_name'] =  img_name
            info_df.iloc[j] = bbox

        info_df['x1'] = info_df['x0'] + info_df['width']
        info_df['y1'] = info_df['y0'] + info_df['height']

        return info_df

def SVHN_get_datasets(data, load_train=True, load_test=True, resize_size=(96, 96)):

    """ Returns SVHN Dataset
    """
    (data_dir, args) = data

    transform = transforms.Compose([transforms.ToTensor(), ai8x.normalize(args=args)])

    if load_train:
        train_dataset = SVHN(root_dir=data_dir, d_type='train',
                             transform=transform, resize_size=resize_size, augment_img=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = SVHN(root_dir=data_dir, d_type='test',
                            transform=transform, resize_size=resize_size)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def SVHN_get_datasets_160_40(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 160x40 images
    """
    return SVHN_get_datasets(data, load_train, load_test, resize_size=(160, 40))

datasets = [
   {
       'name': 'SVHN_160_40',
       'input': (3, 40, 160),
       'output': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
       'loader': SVHN_get_datasets_160_40,
       'collate': SVHN.collate_fn
   },
]