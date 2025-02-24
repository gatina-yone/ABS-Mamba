import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import util.util as util
from models.base_model import BaseModel
from models import networks
from options.test_options import TestOptions
from data import CreateDataLoader
from models.mamba_one import Mamba_model
from models.modules import SAM2Encoder
from util.visualizer import Visualizer
from util import html
import torch
from skimage.metrics import structural_similarity as ssim

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2) 
    if mse == 0:
        return 100  
    return 20 * np.log10(2.0 / np.sqrt(mse))  

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=2)  

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = Mamba_model()
    model.initialize(opt)
    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    psnr_list = []
    ssim_list = []

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        real_A = data['A'].cpu().numpy()  
        real_B = data['B'].cpu().numpy()  

        model.set_input(data)
        model.test()
        fake_B = model.fake_B.cpu().numpy()  
        real_A_numpy = real_A.squeeze()
        fake_B_numpy = fake_B.squeeze()
        real_B_numpy = real_B.squeeze()

        # print(f"real_B min: {real_B_numpy.min()}, max: {real_B_numpy.max()}")
        # print(f"fake_B min: {fake_B_numpy.min()}, max: {fake_B_numpy.max()}")
        # print(f"real_B shape: {real_B_numpy.shape}, fake_B shape: {fake_B_numpy.shape}")  256256
        # print(f"fake_B mean: {fake_B_numpy.mean()}, std: {fake_B_numpy.std()}")
        # print(f"real_B has NaN: {np.isnan(real_B_numpy).any()}, fake_B has NaN: {np.isnan(fake_B_numpy).any()}")
        # print(f"real_B has Inf: {np.isinf(real_B_numpy).any()}, fake_B has Inf: {np.isinf(fake_B_numpy).any()}")

        # print(f"real_A shape: {real_A_numpy.shape}")
        # print(f"fake_B shape: {fake_B_numpy.shape}")
        # print(f"real_B shape: {real_B_numpy.shape}")    

        current_psnr = psnr(fake_B_numpy, real_B_numpy)
        current_ssim = calculate_ssim(fake_B_numpy, real_B_numpy)

        psnr_list.append(current_psnr)
        ssim_list.append(current_ssim)

        print(f'{i:04d}: PSNR = {current_psnr:.2f}, SSIM = {current_ssim:.4f}')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    #test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        if opt.dataset_mode=='aligned_mat':
            visuals=model.get_current_visuals()
            visuals['real_A']=visuals['real_A'][:,:,0:3]
            visuals['real_B']=visuals['real_B'][:,:,0:3]
            visuals['fake_B']=visuals['fake_B'][:,:,0:3]    
            img_path = model.get_image_paths()
            img_path[0]=img_path[0]+str(i)          
        else:
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
    webpage.save()

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f'Average PSNR: {avg_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    txt_path=r'metrics.txt'    
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "a") as f:
        f.write(f"Epoch: {opt.which_epoch}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
