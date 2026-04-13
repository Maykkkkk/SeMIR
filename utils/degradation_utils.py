import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np

from utils.image_utils import crop_img


class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),#patch_size 就是 options.py 里 --patch_size
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):#接收一张干净图 clean_patch，和噪声强度 sigma

        noise = np.random.randn(*clean_patch.shape)#生成一个和 clean_patch 一样大小的随机正态分布噪声
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)#把噪声乘上强度，叠加到干净图上，裁到 0~255 区间，变成无符号 8 位整型。
        return noisy_patch, clean_patch#返回带噪图和原图。

    def _degrade_by_type(self, clean_patch, degrade_type): #根据 degrade_type 确定怎么退化
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch#返回退化后的图和原图。



    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type
        #同时对两张图做退化，方便做训练对比。
        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2#返回降质后的两张图。

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1 #降质一张图
