import os

import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import albumentations as album
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import ai8x


class Watermeter_Roboflow(Dataset):
    def __init__(self, root_dir, d_type, transform=None, resize_size=(96, 96), dials_only=False, augment_img=False):
        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.resize_size = resize_size
        self.dials_only = dials_only

        self.img_list = []
        self.boxes_list = []
        self.lbls_list = []

        self.data_dir = os.path.join(self.root_dir, self.__class__.__name__, self.d_type)

        self.__load_dataset()

        self.g_transforms = None
        self.c_transforms = None
        self.rs_transform = album.Compose(album.Resize(height=resize_size[1], width=resize_size[0]),
                                          bbox_params=album.BboxParams(format='albumentations',
                                                                       label_fields=['class_labels']))
        if augment_img:
            self.c_transforms = album.Compose([album.RandomBrightnessContrast(p=0.75),
                                              #album.RGBShift(r_shift_limit=64, g_shift_limit=64,
                                              #               b_shift_limit=64, p=0.7),
                                              album.ColorJitter(brightness=0.25, contrast=0.25,
                                                                saturation=0.25, hue=0.5, p=0.5),
                                              #album.CLAHE(p=0.7),
                                              album.MultiplicativeNoise(multiplier=(0.15, 0.25), per_channel=True,
                                                                        elementwise=True, p=0.5),
                                              #album.MotionBlur(p=0.7),
                                              album.AdvancedBlur(p=0.5),
                                              #album.InvertImg(p=0.5)
                                              ]
                                             )
            self.g_transforms = album.Compose([
                                               album.Affine(scale=(0.6, 1.5),
                                                            translate_percent=(0.2, 0.4),
                                                            rotate=(-5, 5),
                                                            mode=cv2.BORDER_CONSTANT,
                                                            fit_output=True,
                                                            keep_ratio=True,
                                                            p=0.75),
                                               #album.RandomRotate90(p=0.5)
                                              ],
                                              bbox_params=album.BboxParams(format='albumentations',
                                                                           label_fields=['class_labels']))
            self.rndm_crop = album.Compose([album.RandomCrop(height=resize_size[1], width=resize_size[0])],
                                           bbox_params=album.BboxParams(format='albumentations',
                                                                        label_fields=['class_labels']))



        if dials_only:
            self.rot_transforms = album.Compose([album.RandomRotate90(p=1)],
                                              bbox_params=album.BboxParams(format='albumentations',
                                                                           label_fields=['class_labels']))

    def __load_dataset(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(self.data_dir, filename)
                gt_path = self.__create_gt_path(filename)
                if os.path.isfile(gt_path):
                    self.img_list.append(img_path)
                    boxes, labels = self.__load_gt(gt_path)

                    self.boxes_list.append(np.array(boxes).astype(np.float32))
                    self.lbls_list.append(np.array(labels).astype(np.int64))

    def __create_gt_path(self, filename):
        gtname = filename.replace('.jpg', '.txt')
        return os.path.join(self.data_dir, gtname)

    def __load_gt(self, gt_path):
        #generate boxes in normalized format (x_min, y_min, x_max, y_max)
        labels = []
        boxes = []
        with open(gt_path) as f:
            for line in f:
                vals = line.split()
                label = int(vals[0])
                if label == 0:
                    label = 10
                box = [float(v) for v in vals[1:]]
                box[0] -= 0.5*box[2]
                box[1] -= 0.5*box[3]
                box[2] += box[0]
                box[3] += box[1]

                labels.append(label)
                boxes.append(box)

        return boxes, labels

    def __get_dials_only(self, img, boxes, labels):
        aspect_ratio = self.resize_size[1] / self.resize_size[0]
        margin_ratio = 0.25

        while True:
            x_min = y_min = 1
            x_max = y_max = 0
            for box in boxes:
                x_min = min(x_min, box[0])
                x_max = max(x_max, box[2])
                y_min = min(y_min, box[1])
                y_max = max(y_max, box[3])
            #print('######')

            x_min *= img.shape[1]
            x_max *= img.shape[1]
            y_min *= img.shape[0]
            y_max *= img.shape[0]

            box_width = x_max - x_min
            box_height = y_max - y_min

            if box_width >= box_height:
                expected_height = int(box_height * (1 + margin_ratio))
                if (expected_height / aspect_ratio) < box_width:
                    expected_width = int(box_width * (1 + margin_ratio))
                    expected_height = int(expected_width * aspect_ratio)
                else:
                    expected_width = int(expected_height / aspect_ratio)
                break
            transform_out = self.rot_transforms(image=img, bboxes=boxes, class_labels=labels)
            img = transform_out['image']
            boxes = transform_out['bboxes']
            labels = transform_out['class_labels']
            #plt.imshow(img)
            #plt.show()

        #print(x_max, y_max, expected_width, expected_height)
        x_range = [int(x_max - expected_width), int(x_min)]
        if x_range[1] <= x_range[0]:
            x_range = [int(x_min), int(x_min)+1]
        y_range = [int(y_max - expected_height), int(y_min)]
        if y_range[1] <= y_range[0]:
            y_range = [int(y_min), int(y_min)+1]
        #print(x_range, y_range)

        x0 = np.random.randint(x_range[0], x_range[1])
        x1 = x0 + expected_width
        y0 = np.random.randint(y_range[0], y_range[1])
        y1 = y0 + expected_height

        img_cropped = np.zeros((expected_height, expected_width, 3), np.uint8)

        x_start1 = max(x0, 0)
        x_end1 = min(x1, img.shape[1])
        y_start1 = max(0, y0)
        y_end1 = min(y1, img.shape[0])

        x_start2 = max(0, -x0)
        x_end2 = x_start2 + x_end1 - x_start1
        y_start2 = max(0, -y0)
        y_end2 = y_start2 + y_end1 - y_start1

        img_cropped[y_start2:y_end2, x_start2:x_end2, :] = img[y_start1:y_end1, x_start1:x_end1, :]
        boxes_cropped = []
        for bb in boxes:
            bb = [x_start2 - x_start1 + bb[0]*img.shape[1], y_start2 - y_start1 + bb[1]*img.shape[0],
                  x_start2 - x_start1 + bb[2]*img.shape[1], y_start2 - y_start1 + bb[3]*img.shape[0]]
            bb[0] = max(0, bb[0] / img_cropped.shape[1])
            bb[1] = max(0, bb[1] / img_cropped.shape[0])
            bb[2] = min(1.0, bb[2] / img_cropped.shape[1])
            bb[3] = min(1.0, bb[3] / img_cropped.shape[0])
            boxes_cropped.append(bb)

        return img_cropped, boxes_cropped, labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = self.boxes_list[index]
        labels = self.lbls_list[index]

        if self.g_transforms:
            transform_out = self.g_transforms(image=img, bboxes=boxes, class_labels=labels)
            img = transform_out['image']
            boxes = transform_out['bboxes']
            labels = transform_out['class_labels']

        if self.dials_only:
            #plt.imshow(img)
            #plt.show()
            #print(boxes)
            if self.g_transforms:
                if np.random.rand() < 0.5:
                    img, boxes, labels = self.__get_dials_only(img, boxes, labels)
                else:
                    transform_out = self.rot_transforms(image=img, bboxes=boxes, class_labels=labels)
                    img = transform_out['image']
                    boxes = transform_out['bboxes']
                    labels = transform_out['class_labels']
                    transform_out = self.rndm_crop(image=img, bboxes=boxes, class_labels=labels)
                    img = transform_out['image']
                    boxes = transform_out['bboxes']
                    labels = transform_out['class_labels']
                    # Get a random crop of the image
            else:
                img, boxes, labels = self.__get_dials_only(img, boxes, labels)


        if self.c_transforms:
            transform_out = self.c_transforms(image=img)
            img = transform_out['image']

        transform_out = self.rs_transform(image=img, bboxes=boxes, class_labels=labels)
        img = transform_out['image']
        boxes = transform_out['bboxes']
        labels = transform_out['class_labels']

        if self.transform is not None:
            img = self.transform(img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        #print("img shape: ", img.shape)
        #print("boxes: ", boxes)
        #print("labels: ", labels)

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


def Wtaermeter_RF_get_datasets(data, load_train=True, load_test=True, resize_size=(256, 256), dials_only=True):

    """ Returns Watermeter_Roboflow Dataset
    """
    (data_dir, args) = data

    transform = transforms.Compose([transforms.ToTensor(), ai8x.normalize(args=args)])

    if load_train:
        train_dataset = Watermeter_Roboflow(root_dir=data_dir, d_type='train',
                                            transform=transform, resize_size=resize_size,
                                            dials_only=dials_only, augment_img=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = Watermeter_Roboflow(root_dir=data_dir, d_type='test',
                                           transform=transform, resize_size=resize_size,
                                           dials_only=dials_only)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def Watermeter_RF_get_datasets_160_40(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 160x40 images
    """
    return Wtaermeter_RF_get_datasets(data, load_train, load_test, resize_size=(160, 40))

def Watermeter_RF_get_datasets_clock_256_256(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 160x40 images
    """
    return Wtaermeter_RF_get_datasets(data, load_train, load_test, resize_size=(256, 256), dials_only=False)


datasets = [
   {
       'name': 'Watermeter_RF_160_40',
       'input': (3, 40, 160),
       'output': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
       'loader': Watermeter_RF_get_datasets_160_40,
       'collate': Watermeter_Roboflow.collate_fn
   },

   {
       'name': 'Watermeter_RF_clock_256_256',
       'input': (3, 256, 256),
       'output': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
       'loader': Watermeter_RF_get_datasets_clock_256_256,
       'collate': Watermeter_Roboflow.collate_fn
   },
]