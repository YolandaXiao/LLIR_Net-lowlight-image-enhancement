from os import listdir
from os.path import join
import random
import numpy as np
import cv2

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, decompose_img

#########################################
# Helper functions
#########################################
# get crop positions
def get_params(crop_size, image_size):
    w, h = image_size
    new_h = h
    new_w = w

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))
    return x,y

# get cropped image
def crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

# make the image size power of base
def make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)

# scale the image based on target_size
def scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

#########################################
# Custom dataset class
#########################################
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "low")
        self.b_path = join(image_dir, "high")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        img_low = np.asarray(a)
        img_high = np.asarray(b)

        # decompose images
        kernel = np.ones((5,5),np.uint8)
        I_low, R_low = decompose_img(img_low, kernel)
        I_high, R_high = decompose_img(img_high, kernel)

        # set crop size 
        crop_size = 384
        crop_pos = get_params(crop_size, a.size)

        # define image transform, train image on (384x384) crops
        transform_list = [transforms.Lambda(lambda img: crop(img, crop_pos, crop_size)),
                            transforms.ToTensor()]
        transform_list1 = transform_list + [transforms.Normalize((0.5,), (0.5,))]
        transform_list3 = transform_list + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform_list1)
        transform3 = transforms.Compose(transform_list3)

        # transform illumination and reflectance low high images
        I_low = transform(Image.fromarray(np.uint8(I_low)))
        I_high = transform(Image.fromarray(np.uint8(I_high)))
        R_low = transform3(Image.fromarray(np.uint8(R_low)))
        R_high = transform3(Image.fromarray(np.uint8(R_high)))
        target = transform3(b)

        return I_low, I_high, R_low, R_high, target

    def __len__(self):
        return len(self.image_filenames)


        