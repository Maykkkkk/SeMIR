
import os
import numpy as np
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from src.model import SeMIR
import clip

inputext = [
    "Gaussian noise with a standard deviation of 15",
    "Gaussian noise with a standard deviation of 25",
    "Gaussian noise with a standard deviation of 50",
    "Rain degradation with rain lines",
    "Hazy degradation with normal haze",
    "Blur degradation with motion blur",
    "Lowlight degradation"
]


class SeMIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SeMIR()

    def forward(self, degrad_patch, text_code):
        return self.net(degrad_patch, text_code)



def test_Denoise(net, dataset, sigma=15, text_prompt=""):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(testopt.cuda)
    text_code = net.clip_model.encode_text(text_token).to(dtype=torch.float32)
    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device='cuda', dtype=torch.float32)
            clean_patch = clean_patch.to(device='cuda', dtype=torch.float32)
            text_code = text_code.to(device='cuda', dtype=torch.float32)

            restored = net(degrad_patch, text_code)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain", text_prompt=""):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(testopt.cuda)
    text_code = net.clip_model.encode_text(text_token).to(dtype=torch.float32)
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device='cuda', dtype=torch.float32)
            clean_patch = clean_patch.to(device='cuda', dtype=torch.float32)
            text_code = text_code.to(device='cuda', dtype=torch.float32)

            restored = net(degrad_patch, text_code)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=1,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for enhance, 5 for all-in-one (three tasks), 6 for all-in-one (five tasks)')

    parser.add_argument('--gopro_path', type=str, default="data/test/deblur/", help='save path of test hazy images')
    parser.add_argument('--enhance_path', type=str, default="data/test/enhance/", help='save path of test hazy images')
    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')

    parser.add_argument('--output_path', type=str, default="Output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="", help='checkpoint save path')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    deblur_splits = ["gopro/"]
    enhance_splits = ["lol/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path, i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)
    print("Loading CLIP model...")
    device = f"cuda:{testopt.cuda}"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    print("CKPT name : {}".format(ckpt_path))

    net = SeMIRModel.load_from_checkpoint(ckpt_path).to(device='cuda', dtype=torch.float32)
    net.clip_model = clip_model
    net.eval()

    if testopt.mode == 0:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15, text_prompt=inputext[0])

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25, text_prompt=inputext[1])

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50, text_prompt=inputext[2])


    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain", text_prompt=inputext[3])

    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path, name)
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, derain_set, task="dehaze", text_prompt=inputext[4])

    elif testopt.mode == 3:
        print('Start testing GOPRO...')
        deblur_base_path = testopt.gopro_path
        name = deblur_splits[0]
        testopt.gopro_path = os.path.join(deblur_base_path, name)
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15, task='deblur')
        test_Derain_Dehaze(net, derain_set, task="deblur", text_prompt=inputext[5])

    elif testopt.mode == 4:
        print('Start testing LOL...')
        enhance_base_path = testopt.enhance_path
        name = enhance_splits[0]
        testopt.enhance_path = os.path.join(enhance_base_path, name, task='enhance')
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, derain_set, task="enhance", text_prompt=inputext[6])

    elif testopt.mode == 5:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15, text_prompt=inputext[0])

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25, text_prompt=inputext[1])

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50, text_prompt=inputext[2])

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain", text_prompt=inputext[3])

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze", text_prompt=inputext[4])

    elif testopt.mode == 6:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15, text_prompt=inputext[0])

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25, text_prompt=inputext[1])

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50, text_prompt=inputext[2])

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain", text_prompt=inputext[3])

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze", text_prompt=inputext[4])

        deblur_base_path = testopt.gopro_path
        for name in deblur_splits:
            print('Start testing GOPRO...')

            # print('Start testing {} rain streak removal...'.format(name))
            testopt.gopro_path = os.path.join(deblur_base_path, name)
            deblur_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55, task='deblur')
            test_Derain_Dehaze(net, deblur_set, task="deblur", text_prompt=inputext[5])

        enhance_base_path = testopt.enhance_path
        for name in enhance_splits:
            print('Start testing LOL...')
            testopt.enhance_path = os.path.join(enhance_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55, task='enhance')
            test_Derain_Dehaze(net, derain_set, task="enhance", text_prompt=inputext[6])
