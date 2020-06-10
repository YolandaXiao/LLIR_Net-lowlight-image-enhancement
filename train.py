from __future__ import print_function
import argparse
import os
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, get_scheduler, update_learning_rate, TVloss, Vgg16
from dataset import DatasetFromFolder
from utils import save_img, print_options, init_vgg16, preprocess_vgg16, img_predicted, save_img_np

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--folder_name', required=True, help='name of the folder to save results in')
parser.add_argument('--training_set', type=str, default='our485', help='name of the folder for the trianing set')
parser.add_argument('--eval_set', type=str, default='eval15', help='name of the folder for the evaluation set')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--input_nc_i', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc_i', type=int, default=1, help='output image channels')
parser.add_argument('--input_nc_r', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc_r', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--netG', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | unet_128 ]')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb_i_l1', type=float, default=100, help='weight on L1 term in objective')
parser.add_argument('--lamb_i_l2', type=float, default=100, help='weight on L2 term in objective')
parser.add_argument('--lamb_r_l1', type=float, default=100, help='weight on L1 term in objective')
parser.add_argument('--lamb_r_l2', type=float, default=100, help='weight on L2 term in objective')
parser.add_argument('--lamb_tv', type=float, default=0.0001, help='weight on TV term in objective')
parser.add_argument('--lamb_content', type=float, default=0.0001, help='weight on perceptual term in objective')
parser.add_argument('--lamb_ssim', type=int, default=0, help='weight on ssim term in objective')
parser.add_argument('--lamb_msssim', type=int, default=100, help='weight on ms-ssim term in objective')
opt = parser.parse_args()
print(opt)

# create directories
output_dir = os.path.join("output",opt.folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ckpt_dir = os.path.join(output_dir, "checkpoint")
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
loss_plot_dir = os.path.join(output_dir, "loss_plot")
if not os.path.exists(loss_plot_dir):
    os.makedirs(loss_plot_dir)
eval_dir = os.path.join(output_dir, "train_evaluation")
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
# save opt
print_options(opt, parser, ckpt_dir)


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# load datasets
print('===> Loading datasets')
root_path = "./"
train_set = DatasetFromFolder(root_path + os.path.join(opt.dataset, opt.training_set))
test_set = DatasetFromFolder(root_path + os.path.join(opt.dataset, opt.eval_set))
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)
device = torch.device("cuda:0" if opt.cuda else "cpu")

# Initialize Network
print('===> Building models')
net_i = define_G(opt.input_nc_i, opt.output_nc_i, opt.ngf, opt.netG, 'batch', False, 'normal', 0.02, gpu_id=device)
net_r = define_G(opt.input_nc_r, opt.output_nc_r, opt.ngf, opt.netG, 'batch', False, 'normal', 0.02, gpu_id=device)

# VGG for perceptual loss
if opt.lamb_content > 0:
  vgg = Vgg16()
  init_vgg16(root_path)
  vgg.load_state_dict(torch.load(os.path.join(root_path, "vgg16.weight")))
  vgg.to(device)

# define loss
criterionL1 = nn.L1Loss().to(device)
criterionL2 = nn.MSELoss().to(device)
criterionMSE = nn.MSELoss().to(device)
criterionSSIM = SSIM(data_range=255, size_average=True, channel=3)
criterionMSSSIM1 = MS_SSIM(data_range=255, size_average=True, channel=1)
criterionMSSSIM3 = MS_SSIM(data_range=255, size_average=True, channel=3)

# setup optimizer
optimizer_i = optim.Adam(net_i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_r = optim.Adam(net_r.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_i_scheduler = get_scheduler(optimizer_i, opt)
net_r_scheduler = get_scheduler(optimizer_r, opt)


loss_i_list = []
loss_r_list = []
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    loss_i_per_epoch_list = []
    loss_r_per_epoch_list = []
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        I_low, I_high, R_low, R_high, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device) 
        I_high_rec = net_i(I_low)
        R_high_rec = net_r(R_low)

        # for perceptual loss
        if opt.lamb_content > 0:
          R_high_vgg = preprocess_vgg16(R_high, device)
          R_high_rec_vgg = preprocess_vgg16(R_high_rec, device)
          features_x = vgg(R_high_vgg)
          features_y = vgg(R_high_rec_vgg)

        # unnormalize to calculate ssim
        I_high_rec_tmp = (I_high_rec + 1) / 2
        R_high_rec_tmp = (R_high_rec + 1) / 2
        I_high_tmp = (I_high + 1) / 2
        R_high_tmp = (R_high + 1) / 2
    
        # backward
        optimizer_i.zero_grad()
        optimizer_r.zero_grad()
        # losses
        loss_i_l1 = criterionL1(I_high_rec, I_high) * opt.lamb_i_l1
        loss_r_l1 = criterionL1(R_high_rec, R_high) * opt.lamb_r_l1
        loss_i_l2 = criterionL2(I_high_rec, I_high) * opt.lamb_i_l2
        loss_r_l2 = criterionL2(R_high_rec, R_high) * opt.lamb_r_l2
        loss_tv = TVloss(I_high_rec, opt.lamb_tv)
        loss_i_msssim = (1 - criterionMSSSIM1(I_high_rec_tmp, I_high_tmp)) * opt.lamb_msssim
        loss_r_msssim = (1 - criterionMSSSIM3(R_high_rec_tmp, R_high_tmp)) * opt.lamb_msssim
        # total losses
        loss_i_total = loss_i_l1 + loss_i_l2 + loss_i_msssim + loss_tv
        loss_r_total = loss_r_l1 + loss_r_l2 + loss_r_msssim

        if opt.lamb_content > 0:
            loss_content = criterionL2(features_x[1], features_y[1]) * opt.lamb_content
            loss_r_total += loss_content

        loss_i_per_epoch_list.append(loss_i_total.item())
        loss_r_per_epoch_list.append(loss_r_total.item())
        loss_i_total.backward()
        loss_r_total.backward()
        optimizer_i.step()
        optimizer_r.step()

        print("===> Epoch[{}]({}/{}): loss_i_l1: {:.4f}, loss_r_l1: {:.4f}, loss_i_l2: {:.2f}, loss_r_l2: {:.2f}, loss_i_msssim: {:.4f}, loss_r_msssim: {:.4f}, loss_tv: {:.4f}, loss_content: {:.4f}".format(
              epoch, iteration, len(training_data_loader), loss_i_l1.item(), loss_r_l1.item(), loss_i_l2.item(), loss_r_l2.item(), loss_i_msssim.item(), loss_r_msssim.item(), loss_tv.item(), loss_content.item()))

    update_learning_rate(net_i_scheduler, optimizer_i)
    update_learning_rate(net_r_scheduler, optimizer_r)
    
    loss_i_per_epoch = np.mean(loss_i_per_epoch_list)
    loss_r_per_epoch = np.mean(loss_r_per_epoch_list)
    loss_i_list.append(loss_i_per_epoch)
    loss_r_list.append(loss_r_per_epoch)


    #checkpoint, evalute images, and loss graph
    if epoch % 10 == 0:

        # test
        avg_psnr = 0
        i = 0
        for batch in testing_data_loader:
            # input, target = batch[0].to(device), batch[1].to(device)
            input_i, target_i, input_r, target_r, target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device) 
            prediction_i = net_i(input_i)
            prediction_r = net_r(input_r)
            prediction = prediction_i * prediction_r
            

            # evaluation on images
            prediction, illumination, reflectance = img_predicted(prediction_i, prediction_r)
            save_img_np(prediction, os.path.join(eval_dir, "image{}_epoch{}.jpg".format(str(i), str(epoch))))
            save_img_np(illumination, os.path.join(eval_dir, "image_i{}_epoch{}.jpg".format(str(i), str(epoch))))
            save_img_np(reflectance, os.path.join(eval_dir, "image_r{}_epoch{}.jpg".format(str(i), str(epoch))))
            i+=1

        # save checkpoint
        net_i_model_out_path = os.path.join(ckpt_dir, "netI_model_epoch_{}.pth".format(epoch))
        torch.save(net_i, net_i_model_out_path)
        net_r_model_out_path = os.path.join(ckpt_dir, "netR_model_epoch_{}.pth".format(epoch))
        torch.save(net_r, net_r_model_out_path)
        print("Checkpoint saved to {}".format(ckpt_dir))

        # plot loss graph
        loss_i_plot_path = os.path.join(loss_plot_dir, "loss_i_plot_epoch_{}.jpg".format(epoch))
        loss_r_plot_path = os.path.join(loss_plot_dir, "loss_r_plot_epoch_{}.jpg".format(epoch))
        t = np.arange(epoch)+1
        # illumination
        plt.plot(t, loss_i_list)
        plt.title("Train Loss over Iteration for Illumination")
        plt.ylabel("Train Loss")
        plt.xlabel("Iterations")
        plt.savefig(loss_i_plot_path)
        plt.show()
        # reflectance
        plt.plot(t, loss_r_list)
        plt.title("Train Loss over Iteration for Reflectance")
        plt.ylabel("Train Loss")
        plt.xlabel("Iterations")
        plt.savefig(loss_r_plot_path)
        plt.show()
