import numpy as np
from PIL import Image
import os
import cv2
import math

import torch
# from torch.utils.serialization import load_lua
from torch.autograd import Variable
import torchfile

from networks import Vgg16

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

#########################################
# Image file helper functions
#########################################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def print_options(opt, parser, ckpt_dir):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(ckpt_dir, 'train_opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

#########################################
# VGG Stuff
#########################################
def init_vgg16(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system(
                'wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = torchfile.load(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))

def preprocess_vgg16(batch, device):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]

    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean).to(device)) # subtract mean
    return batch

#########################################
# image helper functions
#########################################
def tensor_to_np(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 
    return image_numpy

def change_range_to_255(img):
    img = img * 255.0
    img = img.clip(0, 255)
    return img.astype(np.uint8)

def img_predicted(prediction_i, prediction_r):
    # get image tensor
    out_i = prediction_i.detach().squeeze(0).cpu()
    out_r = prediction_r.detach().squeeze(0).cpu()
    out_i3 = np.repeat(out_i, 3, axis=0)

    # get image value in numpy [0,1]
    illu_numpy = tensor_to_np(out_i3)
    ref_numpy = tensor_to_np(out_r)

    # compute prediction
    image = change_range_to_255(illu_numpy * ref_numpy)
    illu = change_range_to_255(illu_numpy)
    ref = change_range_to_255(ref_numpy)

    return image, illu, ref

def save_img_np(image_numpy, filename):
    # save image
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def decompose_img(img, kernel):
    # illumination
    I_tmp = np.max(img, axis=-1)
    I = cv2.morphologyEx(I_tmp, cv2.MORPH_CLOSE, kernel)
    I3 = np.repeat(I[..., None], 3, axis=-1)
    
    #reflectance
    eps = 1e-6
    R = img / (I3 + eps) * 255

    return I, R

