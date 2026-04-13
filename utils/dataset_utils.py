import os
import random
import copy
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch
from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


class SeMIRTrainDataset(Dataset):#在train.py中使用，支持所有五种退化类型
    def __init__(self, args):
        super(SeMIRTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []#初始化各种任务的图像id列表
        self.D = Degradation(args)#创建一个图像退化器对象
        self.de_temp = 0
        self.de_type = self.args.de_type#若不指定，de_type默认是 ：'denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance' 这七个
        print(self.de_type)
        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6}#把任务名映射成编号
        self._init_ids()#初始化各任务的图像路径，调用后面的 _init_clean_ids()、_init_rs_ids()……等方法。
        self._merge_ids()# 把所有子任务的图片路径和任务编号合并到一个大列表里。
        self.crop_transform = Compose([  #定义随机裁剪和Tensor转换
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

    def _init_ids(self): #_init_ids() — 各任务路径初始化调度器,检查当前需要训练哪些任务，调用对应的 self._init_xxx_ids() 方法。
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()

        random.shuffle(self.de_type)
        #多了deblur、enhance



    def _init_clean_ids(self):#根据 denoise.txt 的列表筛选去噪图像目录里的文件，生成完整路径列表，后续就可以用这个列表去加载你要处理的图像了。
        ref_file = self.args.data_file_dir + "noisy/denoise.txt" #路径为：data_file_dir=data_dir/, 所以完整路径是：data_dir/noisy/denoise.txt
        temp_ids = []#创建一个空的列表
        temp_ids+= [id_.strip() for id_ in open(ref_file)] #遍历data_dir/noisy/denoise.txt中每一行内容，对每行执行 .strip() ，去掉首尾空白字符（空格、换行符、制表符等）， 然后把结果放进一个新列表里
        clean_ids = []#创建空列表
        name_list = os.listdir(self.args.denoise_dir)#使用 os.listdir(args.denoise_dir) 获取去噪数据目录（args.denoise_dir='data/Train/Denoise/'）中的所有文件名（name_list）
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]#把 name_list 中出现在 temp_ids 里的 id，拼接上 denoise 路径，组成完整路径，追加到 clean_ids 列表里。

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]#用列表推导式，为clean_ids中的每个图像路径创建一个字典，存入self.s15_ids
            self.s15_ids = self.s15_ids * 3#将 self.s15_ids 重复 3 次，增加数据量。目的是平衡去噪任务与其他任务（如去雾的 72,135 张）的数据量，防止模型偏向数据量大的任务。
            random.shuffle(self.s15_ids)#随机打乱
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]#将clean_ids中的每个元素x都生成一个字典：{"clean_id": x,"de_type":1}，包含两个键值对
            self.s25_ids = self.s25_ids * 3#将 self.s25_ids 重复 3 次，增加数据量。例如，若 clean_ids 有 5,144 张图像，s25_ids 从 5,144 扩展到 5,144 × 3 = 15,432 张。
            random.shuffle(self.s25_ids)#随机打乱 self.s25_ids
            self.s25_counter = 0#可能是为分批加载、样本跟踪或其他功能预留。
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)#干净图像数
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = [] #创建空列表
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt" #路径为：data_dir/hazy/hazy_outside.txt
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]#self.args.dehaze_dir=data/Train/Dehaze/ ; 遍历hazy文件下的每一行，去掉每行字符串前后的空格和换行符。然后拼接上路径data/Train/Dehaze/，最后装进列表temp_ids里面
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_deblur_ids(self):#根据 denoise.txt 的列表筛选去噪图像目录里的文件，生成完整路径列表，后续就可以用这个列表去加载你要处理的图像了。
        temp_ids = []

        image_list = os.listdir(os.path.join(self.args.gopro_dir, 'blur/'))#data/Train/Deblur/blur/
        temp_ids = image_list #路径为：data/Train/Deblur/blur/
        self.deblur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        self.deblur_ids = self.deblur_ids * 5 # GOPro中2,103张用于训练 ， 5倍后为10515
        self.deblur_counter = 0
        self.num_deblur = len(self.deblur_ids)
        print('Total Blur Ids : {}'.format(self.num_deblur))

    def _init_enhance_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.enhance_dir, 'low/'))#data/Train/Enhance/low/
        temp_ids = image_list
        self.enhance_ids= [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.enhance_ids = self.enhance_ids * 20 #重复20次增加数据量 ， LOL-V1 共485 张 ，乘20等于 9700张
        self.num_enhance = len(self.enhance_ids)
        print('Total enhance Ids : {}'.format(self.num_enhance))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"#data_dir/rainy/rainTrain.txt
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]#data/Train/Derain/ 加上 rainTrain.txt中去掉空格等等的图像文件名（data/Train/Derain/rainy/rain-100.png）放进列表 temp_ids 中
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120 # 去雨的数据集Rain100L有200张图，乘20后 共 24000 张

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2): #开始裁剪
        H = img_1.shape[0]#获取输入图像宽高
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)#ind_H：高度方向的起始索引，范围 [0, H - patch_size]。
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]#patch_1：从 img_1 裁剪出 (ind_H:ind_H+128, ind_W:ind_W+128) 的区域，形状为 (128, 128, 3)。
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]#patch_2：从 img_2 裁剪出相同区域，形状相同。确保 patch_1 和 patch_2 位置对齐（如退化图像和干净图像的对应区域）。

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]#用于根据去雨任务的退化图像路径（rainy_name）生成对应的干净图像路径（gt_name）
        return gt_name


    def _get_deblur_name(self, deblur_name):#
        gt_name = deblur_name.replace("blur", "sharp")#根据模糊图像（deblur_name）的路径生成对应的清晰图像（ground truth, GT）路径。replace("blur", "sharp") 将路径中的 "blur" 子字符串替换为 "sharp"。
        return gt_name
    

    def _get_enhance_name(self, enhance_name):
        gt_name = enhance_name.replace("low", "gt")#将路径中的 "low" 子字符串替换为 "gt"。
        return gt_name


    def _get_nonhazy_name(self, hazy_name):#根据有雾图像（hazy_name）的路径生成对应的无雾图像（ground truth, GT）路径
        dir_name = hazy_name.split("synthetic")[0] + 'original/'#用 split("synthetic")[0] 获取 "synthetic" 前的路径前缀，添加 "original/"，生成无雾图像目录: data/Train/Dehaze/original/
        name = hazy_name.split('/')[-1].split('_')[0]# split('/')[-1]：按/来分隔，取最后一个 ； split('_')[0]：按_来分隔，取第一个 , 比如 data/Train/Dehaze/synthetic/part1/0025_0.8_0.1.jpg -> 0025
        suffix = '.' + hazy_name.split('.')[-1] # 按 . 进行分隔，取最后一个（即jpg），最后得到： .jpg
        nonhazy_name = dir_name + name + suffix #data/Train/Dehaze/synthetic/part1/0025_0.8_0.1.jpg  -> data/Train/Dehaze/original/0025.jpg
        return nonhazy_name #返回原始图像的名称

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#AirNet没有此函数
    def _merge_ids(self):
        self.sample_ids = [] #初始化空列表
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.deblur_ids
        if "enhance" in self.de_type:
            self.sample_ids += self.enhance_ids

        print(len(self.sample_ids))
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        sample = self.sample_ids[idx]#根据给定的索引 idx，取出列表中的第 idx 个元素，赋值给 sample。其中 self.sample_ids 是一个列表，存储所有任务（去噪、去雨、去雾等）的图像路径和任务类型
        de_id = sample["de_type"]
        if de_id < 3:#de_id < 3（去噪任务：denoise_15, denoise_25, denoise_50）
            if de_id == 0:
                clean_id = sample["clean_id"]#返回字典中的图像路径
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)#将clean_patch转换为 numpy.ndarray 类型

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id) #single_degrade：Degradation 类的方法，接收干净图像 patch（clean_patch）和退化类型编号（de_id），返回一个 degrad_patch_1 给 degrad_patch
        else:
            if de_id == 3:  #因为'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)#sample 是从 self.sample_ids[idx] 获取的字典，格式如 {"clean_id": path, "de_type": id}。
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # Deblur with Gopro set
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.gopro_dir, 'blur/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.gopro_dir, 'sharp/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = self._get_deblur_name(sample["clean_id"])
            elif de_id == 6:
                # Enhancement with LOL training set
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.enhance_dir, 'low/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.enhance_dir, 'gt/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = self._get_enhance_name(sample["clean_id"])

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        clean_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch#这里只返回clean_patch、degrade_patch，是因为该任务不像AirNet一样需要来自同一张图片裁剪的两个patch去作为输入

    def __len__(self):
        return len(self.sample_ids)#self.sample_ids 是一个列表，存储参数--de_type指定的的图像路径和任务类型，由 _merge_ids 方法生成。len(self.sample_ids) 计算列表长度，即数据集的样本总数。


class DenoiseTestDataset(Dataset): # 去噪任务测试集，通过动态添加高斯噪声生成退化图像。
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)#从 args.denoise_path：data/test/denoise/bsd68  读取文件名，生成干净图像路径列表 clean_ids
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)#为干净图像 clean_patch 添加高斯噪声（强度 self.sigma）
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma#设置噪声强度

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #没有被使用，故注释掉
    # def tile_degrad(input_,tile=128,tile_overlap =0):
    #     sigma_dict = {0:0,1:15,2:25,3:50}
    #     b, c, h, w = input_.shape
    #     tile = min(tile, h, w)
    #     assert tile % 8 == 0, "tile size should be multiple of 8"
    #
    #     stride = tile - tile_overlap
    #     h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    #     w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    #     E = torch.zeros(b, c, h, w).type_as(input_)
    #     W = torch.zeros_like(E)
    #     s = 0
    #     for h_idx in h_idx_list:
    #         for w_idx in w_idx_list:
    #             in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
    #             out_patch = in_patch
    #             out_patch_mask = torch.ones_like(in_patch)
    #
    #             E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
    #             W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    #     restored = torch.clamp(restored, 0, 1)
    #     return restored

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur': 2, 'enhance': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)

    """_add_gaussian_nois函数未使用，因为在测试脚本中所有addnoise都为False，故不会调用此函数，因此不论传入的sigma是多少也都用不上，忽略它"""
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.gopro_path +'input/')
            self.ids += [self.args.gopro_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            name_list = os.listdir(self.args.enhance_path + 'input/')
            self.ids += [self.args.enhance_path + 'input/' + id_ for id_ in name_list]


        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target").replace("rain-", "norain-")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "target")

        elif self.task_idx == 3:
            gt_name = degraded_name.replace("input", "target")

        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:#TODO  这里为啥有加噪的需要？ 引用的文件中的所有传入参数都是： addnoise=False
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length

#未使用，故注释
# class TestSpecificDataset(Dataset):
#     def __init__(self, args):
#         super(TestSpecificDataset, self).__init__()
#         self.args = args
#         self.degraded_ids = []
#         self._init_clean_ids(args.test_path)
#
#         self.toTensor = ToTensor()
#
#     def _init_clean_ids(self, root):
#         extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
#         if os.path.isdir(root):
#             name_list = []
#             for image_file in os.listdir(root):
#                 if any([image_file.endswith(ext) for ext in extensions]):
#                     name_list.append(image_file)
#             if len(name_list) == 0:
#                 raise Exception('The input directory does not contain any image files')
#             self.degraded_ids += [root + id_ for id_ in name_list]
#         else:
#             if any([root.endswith(ext) for ext in extensions]):
#                 name_list = [root]
#             else:
#                 raise Exception('Please pass an Image file')
#             self.degraded_ids = name_list
#         print("Total Images : {}".format(name_list))
#
#         self.num_img = len(self.degraded_ids)
#
#     def __getitem__(self, idx):
#         degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
#         name = self.degraded_ids[idx].split('/')[-1][:-4]
#
#         degraded_img = self.toTensor(degraded_img)
#
#         return [name], degraded_img
#
#     def __len__(self):
#         return self.num_img
#
