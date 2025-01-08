import os

import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import albumentations as album
import albumentations.augmentations.crops.functional as cf
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms


from math import atan2
import pickle

import ai8x


class Watermeter_Kaggle(Dataset):
    def __init__(self, root_dir, d_type, transform=None, resize_size=(96, 96), augment_img=False):
        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.resize_size = resize_size

        self.img_list = []
        self.mask_list = []
        self.boxes_list = []
        self.lbls_list = []
        self.kpts_list = []
        if not os.path.exists(self.root_dir):
            raise ValueError("Dataset does not exist")

        if not os.path.exists(os.path.join(self.root_dir, self.__class__.__name__, self.d_type)):
            self.split_data()

        self.data_dir = os.path.join(self.root_dir, self.__class__.__name__, self.d_type)


        self.gt_path = os.path.join(self.data_dir, 'gt.pickle')



        if not os.path.exists(self.gt_path):
           self.__load_dataset()
           self.__generate_gt()

        with open(self.gt_path, 'rb') as f:
            gt = pickle.load(f)
            self.boxes_list = gt[0]
            self.lbls_list = gt[1]
            self.kpts_list = gt[2]
            self.img_list = gt[3]
            #print("self.img_list: ", self.img_list)
            self.mask_list = gt[4]

        print("Number of images: ", len(self.img_list))
        print("Number of masks: ", len(self.mask_list))

        self.g_transforms = None
        self.c_transforms = None
        #self.rs_transform = album.Compose(
        #                                  album.Resize(height=resize_size[1], width=resize_size[0]),
        #                                  bbox_params=album.BboxParams(format='albumentations',
        #                                                               label_fields=['class_labels']),
        #                                    keypoint_params=album.KeypointParams(format='xy',
        #                                                       remove_invisible=False))
        if augment_img:
            self.rs_transform = album.Compose(
                                          [album.BBoxSafeRandomCrop(erasing_rate=0.3),
                                           album.Resize(height=resize_size[1], width=resize_size[0])],
                                            bbox_params=album.BboxParams(format='albumentations',
                                                                          label_fields=['class_labels']),
                                            keypoint_params=album.KeypointParams(format='xy',
                                                                remove_invisible=False)
                                        )
            self.c_transforms = album.Compose([album.RandomBrightnessContrast(p=0.75),
                                              album.RGBShift(r_shift_limit=32, g_shift_limit=32,
                                                             b_shift_limit=32, p=0.5),
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
                                               album.Affine(
                                                            translate_percent=(0.2, 0.4),
                                                            rotate=(-45, 45),
                                                            mode=cv2.BORDER_CONSTANT,
                                                            fit_output=False,
                                                            keep_ratio=True,
                                                            p=0.75),
                                               #album.RandomRotate90(p=0.5)
                                              ],
                                              bbox_params=album.BboxParams(format='albumentations',
                                                                           label_fields=['class_labels']),
                                              keypoint_params=album.KeypointParams(format='xy',
                                                               remove_invisible=False))
        else:
            self.rs_transform = album.Compose(
                                          album.Resize(height=resize_size[1], width=resize_size[0]),
                                          bbox_params=album.BboxParams(format='albumentations',
                                                                       label_fields=['class_labels']),
                                            keypoint_params=album.KeypointParams(format='xy',
                                                               remove_invisible=False))

    def split_data(self):
        data_dir = os.path.join(self.root_dir, self.__class__.__name__)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        images_path = os.path.join(data_dir, 'images')
        masks_path = os.path.join(data_dir, 'masks')

        for filename in os.listdir(images_path):
            if filename.endswith('.jpg'):
                img_path = os.path.join(images_path, filename)
                mask_path = os.path.join(masks_path, filename)
                if np.random.rand() < 0.9:
                    os.rename(img_path, os.path.join(train_dir, filename))
                    os.rename(mask_path, os.path.join(train_dir, filename.replace('.jpg', '_mask.jpg')))
                else:
                    os.rename(img_path, os.path.join(test_dir, filename))
                    os.rename(mask_path, os.path.join(test_dir, filename.replace('.jpg', '_mask.jpg')))


    def __load_dataset(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.jpg') and not filename.endswith('_mask.jpg'):
                self.img_list.append(os.path.join(self.data_dir, filename))
                mask_file = os.path.join(self.data_dir, filename.replace('.jpg', '_mask.jpg'))
                self.mask_list.append(mask_file)

    @staticmethod
    def __getOrientation(pts):
        ## [pca]
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]

        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, _ = cv2.PCACompute2(data_pts, mean)

        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))
        ## [pca]

        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

        return cntr, angle


    def __generate_gt(self):

        boxes_list = []
        lbls_list = []
        kpts_list = []
        img_list_revised = []
        mask_list_revised = []
        abs_path = os.path.abspath(self.data_dir)
        for filename in self.img_list:
            print("Processing image: ", filename)
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_file = filename.replace('.jpg', '_mask.jpg')
            mask = cv2.imread(mask_file) #, cv2.IMREAD_GRAYSCALE) TODO: can be used as well??
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            boxes = []
            labels = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    center, angle = self.__getOrientation(contour)
                    if angle < -1 or angle > 1:
                        #radians to degrees
                        rotation_angle = angle * 180 / np.pi
                        M = cv2.getRotationMatrix2D(center, rotation_angle, 1)
                        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 1000:
                                center, angle = self.__getOrientation(contour)
                                break
                        #save the image
                        filename = filename.replace('.jpg', '_rotated.jpg')
                        mask_file = mask_file.replace('_mask.jpg', '_rotated_mask.jpg')
                        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(mask_file, mask)

                    x, y, w, h = cv2.boundingRect(contour)
                    #print("x, y, w, h: ", x, y, w, h)
                    min_rect = cv2.minAreaRect(contour)
                    min_corners = cv2.boxPoints(min_rect).tolist()
                    #sort the corners
                    min_corners = sorted(min_corners, key=lambda x: x[0], reverse=True)
                    min_corners[:2] = sorted(min_corners[:2], key=lambda x: x[1], reverse=True)
                    min_corners[2:] = sorted(min_corners[2:], key=lambda x: x[1], reverse=True)
                    #print("min_corners: ", min_corners)
                    # normalize x, y, w, h
                    x = x / img.shape[1]
                    y = y / img.shape[0]
                    w = w / img.shape[1]
                    h = h / img.shape[0]
                    boxes.append([x, y, x+w, y+h])
                    labels.append(1)
                    # normalize keypoints
                    #for min_corner in min_corners:
                    #    min_corner[0] = min_corner[0] / img.shape[1]
                    #    min_corner[1] = min_corner[1] / img.shape[0]

                    #print("min_corners: ", min_corners)
                    #print("x, y, w, h: ", x, y, w, h)
                    #print("boxes: ", boxes)
                    #print("labels: ", labels)
                    #print("area: ", area)
                    #print("Img name: ", filename)

                    #print("center: ", center)
            if len(boxes) == 0:
                print("No boxes found for image: ", filename)
                continue
            boxes_list.append(boxes)
            lbls_list.append(labels)
            kpts_list.append(min_corners)
            filename = os.path.basename(filename)
            mask_file = os.path.basename(mask_file)
            img_list_revised.append(os.path.join(abs_path, filename))
            mask_list_revised.append(os.path.join(abs_path, mask_file))

        with open(self.gt_path, 'wb') as f:
            pickle.dump((boxes_list, lbls_list, kpts_list, img_list_revised, mask_list_revised), f)

    def __clamp_normalized_kpts(self, box):
        return np.clip(box, 0., 1.)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = self.boxes_list[index]
        labels = self.lbls_list[index]
        keypoints = self.kpts_list[index]





        #print("boxes: ", boxes)
        #print("labels: ", labels)
        #print("keypoints: ", keypoints)
        #print("img shape: ", img.shape)
        if self.g_transforms:
             # Random Slighlty increase the size of the bounding box
            if np.random.rand() < 0.5:
                random_enlarge = np.random.rand() * 0.03
                for box_idx, box in enumerate(boxes):
                    x, y, x1, y1 = box
                    x = x - random_enlarge
                    y = y - random_enlarge
                    x1 = x1 + random_enlarge
                    y1 = y1 + random_enlarge
                    boxes[box_idx] = np.clip([x, y, x1, y1], 0., 1.)

            transform_out = self.g_transforms(image=img, bboxes=boxes, keypoints=keypoints, class_labels=labels)
            img = transform_out['image']
            boxes = transform_out['bboxes']
            keypoints = transform_out['keypoints']
            labels = transform_out['class_labels']


        if self.c_transforms:
            transform_out = self.c_transforms(image=img)
            img = transform_out['image']

        #print("boxes: ", boxes)
        #print("labels: ", labels)
        #print("keypoints: ", keypoints)
        if not boxes:
            boxes = [[0., 0., 1., 1.]]
            labels = [0]
            keypoints = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        

        #Crop the ROI
        crop_cords = [int(boxes[0][0] * img.shape[1]), int(boxes[0][1] * img.shape[0]),
                      int(boxes[0][2] * img.shape[1]), int(boxes[0][3] * img.shape[0])]

        crop_cords[0] = max(0, crop_cords[0] - int(0.05 * img.shape[1]))
        crop_cords[1] = max(0, crop_cords[1] - int(0.1 * img.shape[0]))
        crop_cords[2] = min(img.shape[1], crop_cords[2] + int(0.05 * img.shape[1]))
        crop_cords[3] = min(img.shape[0], crop_cords[3] + int(0.1 * img.shape[0]))
        #print("Before crop", img.shape)
        boxes = cf.crop_bboxes_by_coords(np.array(boxes), crop_cords, (img.shape[0], img.shape[1]))
        keypoints = cf.crop_keypoints_by_coords(np.array(keypoints), crop_cords)
        img = img[crop_cords[1]:crop_cords[3], crop_cords[0]:crop_cords[2]]
        #print("After crop", img.shape)

        
        transform_out = self.rs_transform(image=img, bboxes=boxes, keypoints=keypoints, class_labels=labels)
        img = transform_out['image']
        boxes = transform_out['bboxes']
        keypoints = transform_out['keypoints']
        #print("boxes: ", boxes)
        #print("keypoints: ", keypoints)
        #print("img shape: ", img.shape)
        # Normalize keypoints:
        
        
        for kp_idx, keypoint in enumerate(keypoints):
            keypoints[kp_idx] = self.__clamp_normalized_kpts(
                                        [float(keypoint[0]/img.shape[1]),
                                        float(keypoint[1]/img.shape[0])])


        for kp_idx, keypoint in enumerate(keypoints):
            keypoints[kp_idx] = self.__clamp_normalized_kpts([(keypoint[0] - boxes[0][0]) / (boxes[0][2] - boxes[0][0]),
                                (keypoint[1] - boxes[0][1]) / (boxes[0][3] - boxes[0][1])])




        #print("keypoints aft2: ", keypoints)
        labels = transform_out['class_labels']

        if self.transform is not None:
            img = self.transform(img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        #print("img shape: ", img.shape)
        #print("boxes: ", boxes)
        #print("labels: ", labels)
        #print("keypoints: ", keypoints)

        return img, (boxes, keypoints, labels)

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



class Watermeter_Kaggle_Segmentation(Dataset):
    def __init__(self, root_dir, d_type, transform=None, resize_size=(96, 96), augment_img=False):
        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.resize_size = resize_size

        self.img_list = []
        self.mask_list = []
        self.mask_fold = ai8x.fold(fold_ratio=4)

        if not os.path.exists(self.root_dir):
            raise ValueError("Dataset does not exist")

        if not os.path.exists(os.path.join(self.root_dir, self.__class__.__name__, self.d_type)):
            self.split_data()

        self.data_dir = os.path.join(self.root_dir, self.__class__.__name__, self.d_type)


        self.gt_path = os.path.join(self.data_dir, 'gt.pickle')



        if not os.path.exists(self.gt_path):
           self.__load_dataset()
           self.__generate_gt()

        with open(self.gt_path, 'rb') as f:
            gt = pickle.load(f)

            self.img_list = gt[0]
            #print("self.img_list: ", self.img_list)
            self.mask_list = gt[1]

        print("Number of images: ", len(self.img_list))
        print("Number of masks: ", len(self.mask_list))

        self.g_transforms = None
        self.c_transforms = None

        if augment_img:
            self.rs_transform = album.Compose(
                                          album.RandomResizedCrop(height=resize_size[1], width=resize_size[0]),
                                        )

            self.c_transforms = album.Compose([album.RandomBrightnessContrast(p=0.75),
                                              album.RGBShift(r_shift_limit=32, g_shift_limit=32,
                                                             b_shift_limit=32, p=0.5),
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
                                               album.Affine(
                                                            translate_percent=(0.2, 0.4),
                                                            rotate=(-45, 45),
                                                            mode=cv2.BORDER_CONSTANT,
                                                            fit_output=False,
                                                            keep_ratio=True,
                                                            p=0.75),
                                               #album.RandomRotate90(p=0.5)
                                              ],
                                              )
        else:
            self.rs_transform = album.Compose(
                                          album.Resize(height=resize_size[1], width=resize_size[0]),
                                        )

    def split_data(self):
        data_dir = os.path.join(self.root_dir, self.__class__.__name__)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        images_path = os.path.join(data_dir, 'images')
        masks_path = os.path.join(data_dir, 'masks')

        for filename in os.listdir(images_path):
            if filename.endswith('.jpg'):
                img_path = os.path.join(images_path, filename)
                mask_path = os.path.join(masks_path, filename)
                if np.random.rand() < 0.9:
                    os.rename(img_path, os.path.join(train_dir, filename))
                    os.rename(mask_path, os.path.join(train_dir, filename.replace('.jpg', '_mask.jpg')))
                else:
                    os.rename(img_path, os.path.join(test_dir, filename))
                    os.rename(mask_path, os.path.join(test_dir, filename.replace('.jpg', '_mask.jpg')))


    def __load_dataset(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.jpg') and not filename.endswith('_mask.jpg'):
                self.img_list.append(os.path.join(self.data_dir, filename))
                mask_file = os.path.join(self.data_dir, filename.replace('.jpg', '_mask.jpg'))
                self.mask_list.append(mask_file)

    def __generate_gt(self):

        img_list_revised = []
        mask_list_revised = []
        abs_path = os.path.abspath(self.data_dir)
        for filename in self.img_list:
            print("Processing image: ", filename)
            mask_file = filename.replace('.jpg', '_mask.jpg')
            mask = cv2.imread(mask_file) #, cv2.IMREAD_GRAYSCALE) TODO: can be used as well??
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

            filename = os.path.basename(filename)

            img_list_revised.append(os.path.join(abs_path, filename))
            mask_list_revised.append(mask)

        with open(self.gt_path, 'wb') as f:
            pickle.dump((img_list_revised, mask_list_revised), f)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.mask_list[index]





        #print("boxes: ", boxes)
        #print("labels: ", labels)
        #print("keypoints: ", keypoints)
        #print("img shape: ", img.shape)
        if self.g_transforms:


            transform_out = self.g_transforms(image=img, mask=mask)
            img = transform_out['image']
            mask = transform_out['mask']


        if self.c_transforms:
            transform_out = self.c_transforms(image=img)
            img = transform_out['image']


        #print("After crop", img.shape)
        transform_out = self.rs_transform(image=img, mask=mask)
        img = transform_out['image']
        mask = transform_out['mask']
        #print("boxes: ", boxes)
        #print("keypoints: ", keypoints)
        #print("img shape: ", img.shape)
        # Normalize keypoints:


        if self.transform is not None:
            img = self.transform(img)

        mask = mask / 255
        mask = torch.as_tensor(mask, dtype=torch.int64)


        #print("img shape: ", img.shape)
        #print("boxes: ", boxes)
        #print("labels: ", labels)
        #print("keypoints: ", keypoints)

        return img, mask



def Watermeter_Kaggle_get_datasets(data, load_train=True, load_test=True, resize_size=(256, 256)):

    """ Returns Watermeter_Kaggle Dataset
    """
    (data_dir, args) = data

    transform = transforms.Compose([transforms.ToTensor(), ai8x.normalize(args=args)])

    if load_train:
        train_dataset = Watermeter_Kaggle(root_dir=data_dir, d_type='train',
                                            transform=transform, resize_size=resize_size,
                                            augment_img=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = Watermeter_Kaggle(root_dir=data_dir, d_type='test',
                                           transform=transform, resize_size=resize_size,
                                           )
    else:
        test_dataset = None

    return train_dataset, test_dataset


def Watermeter_Kaggle_segmentation_get_datasets(data, load_train=True, load_test=True, resize_size=(256, 256)):
    """ Returns Watermeter_Kaggle Dataset
    """
    (data_dir, args) = data

    transform = transforms.Compose([transforms.ToTensor(), ai8x.normalize(args=args), ai8x.fold(fold_ratio=4)])

    if load_train:
        train_dataset = Watermeter_Kaggle_Segmentation(root_dir=data_dir, d_type='train',
                                            transform=transform, resize_size=resize_size,
                                            augment_img=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = Watermeter_Kaggle_Segmentation(root_dir=data_dir, d_type='test',
                                           transform=transform, resize_size=resize_size,
                                           )
    else:
        test_dataset = None

    return train_dataset, test_dataset


def Watermeter_Kaggle_get_datasets_clock_256_256(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 160x40 images
    """
    return Watermeter_Kaggle_get_datasets(data, load_train, load_test, resize_size=(256, 256))

def Watermeter_Kaggle_get_datasets_clock_256_256_segmentation(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 160x40 images
    """
    return Watermeter_Kaggle_segmentation_get_datasets(data, load_train, load_test, resize_size=(256, 256))


datasets = [
   {
       'name': 'Watermeter_Kaggle_clock_256_256',
       'input': (3, 256, 256),
       'output': ([1]),
       'loader': Watermeter_Kaggle_get_datasets_clock_256_256,
       'collate': Watermeter_Kaggle.collate_fn
   },
   {
       'name': 'Watermeter_Kaggle_clock_256_256_segmentation',
       'input': (48, 64, 64),
       'output': (0, 1),
       'loader': Watermeter_Kaggle_get_datasets_clock_256_256_segmentation,
       'fold_ratio': 4,
   },
]