from __future__ import print_function
import argparse
import numpy as np
import os
import time
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img, decompose_img, save_img_np, img_predicted, calculate_psnr
from dataset import scale_width, make_power_2

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--folder_name', required=True, help='name of the folder to run results')
parser.add_argument('--dataset_gt', type=str, help='facades')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--result_path', type=str, default='result', help='notes on the run')
parser.add_argument('--image_width', type=int, default=768, help='notes on the run')
opt = parser.parse_args()
print(opt)
device = torch.device("cuda:0" if opt.cuda else "cpu")

# model paths
output_dir = os.path.join("output", opt.folder_name)
ckpt_dir = os.path.join(output_dir, "checkpoint")
result_dir = os.path.join(output_dir, opt.result_path)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

model_r_path = os.path.join(ckpt_dir, "netR_model_epoch_{}.pth".format(str(opt.nepochs)))
model_i_path = os.path.join(ckpt_dir, "netI_model_epoch_{}.pth".format(str(opt.nepochs)))
# if cpu
net_r = torch.load(model_r_path, map_location={'cuda:0': 'cpu'}).to(device)
net_i = torch.load(model_i_path, map_location={'cuda:0': 'cpu'}).to(device)

# image paths
image_dir = opt.dataset
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]


# image transformation
transform_list = [ transforms.Lambda(lambda img: scale_width(img, opt.image_width, 128, method=Image.BICUBIC)),
            transforms.Lambda(lambda img: make_power_2(img, 128, method=Image.BICUBIC)),
            transforms.ToTensor()]
transform_list1 = transform_list + [transforms.Normalize((0.5,), (0.5,))]
transform_list3 = transform_list + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list1)
transform3 = transforms.Compose(transform_list3)

avg_psnr = 0
avg_ssim = 0
avg_time = 0
count = 0
for image_name in image_filenames:
    img = load_img(image_dir + image_name)

    # get illumination and reflectance
    kernel = np.ones((5,5),np.uint8)
    I_low, R_low = decompose_img(img, kernel)
    I_low = transform(Image.fromarray(np.uint8(I_low)))
    R_low = transform3(Image.fromarray(np.uint8(R_low)))

    # run them through the network
    input_i = I_low.unsqueeze(0).to(device)
    input_r = R_low.unsqueeze(0).to(device)
    prediction_i = net_i(input_i)
    prediction_r = net_r(input_r)

    # get results
    prediction, illumination, reflectance = img_predicted(prediction_i, prediction_r)

    # save image
    image_id = image_name.split('.')[0]
    save_img_np(prediction, os.path.join(result_dir, "{}_epoch{}.jpg".format(image_id, str(opt.nepochs))))
    save_img_np(illumination, os.path.join(result_dir, "{}_illumination.jpg".format(image_id)))
    save_img_np(reflectance, os.path.join(result_dir, "{}_reflectance.jpg".format(image_id)))

    # compute evaluation
    if opt.dataset_gt:
        img_gt = load_img(opt.dataset_gt + image_name)
        img_gt = scale_width(img_gt, opt.image_width, 256, method=Image.BICUBIC)
        img_gt = make_power_2(img_gt, 256, method=Image.BICUBIC)
        target = np.asarray(img_gt)

        psnr_eval = peak_signal_noise_ratio(target, prediction)
        ssim_eval = structural_similarity(target, prediction, multichannel=True)

        avg_psnr += psnr_eval 
        avg_ssim += ssim_eval
        count += 1

if opt.dataset_gt: 
    print("===> Avg. PSNR1: {:.4f} dB, Avg. SSIM: {:.4f}".format(avg_psnr1/count, avg_ssim/count))



