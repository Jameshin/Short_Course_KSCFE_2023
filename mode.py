=====================
Edited by Sajid @ UST
=====================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from dataset import *
from srgan_model import Generator, Discriminator, GeneratorS
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio

from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
def psnr(output, target, max_val=1.0):
    mse = torch.mean((output - target) ** 2)
    return 10 * torch.log10((max_val ** 2) / mse)
    
import horovod.torch as hvd

def train(args):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #cuda = torch.cuda.is_available()
    print(device)
    transform  = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    

    ################
    # Models Loader
    ################

    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num, scale=args.scale)
    generatorS = GeneratorS(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 4, scale=args.scale)       
    discriminator = Discriminator(patch_size = args.patch_size * args.scale)
    
    if args.fine_tuning:        
        generatorS.load_state_dict(torch.load(args.generator_pathS))
        print("pre-trained student model model is loaded")
        print("path : %s"%(args.generator_pathS))
        
        discriminator.load_state_dict(torch.load(args.discriminator_path))
        print("pre-trained discriminator model is loaded")
        print("path : %s"%(args.discriminator_path))
    
    #discriminator = discriminator.to(device)
    discriminator.train()
   
    generator.load_state_dict(torch.load(args.generator_path))
    print("pre-trained Teacher model is loaded")
    print("path : %s"%(args.generator_path))
        
    #generator = generator.to(device)
    generator.eval()

    #generatorS = generatorS.to(device)
    generatorS.train()
    
    #### Train using perceptual & adversarial loss
    vgg_net = vgg19()
    vgg_net = vgg_net.eval()
    
    
    ###################
    # Loss Functions
    ###################
    
    l1_loss = torch.nn.L1Loss() 
    l2_loss = nn.MSELoss()
    VGG_loss = perceptual_loss(vgg_net) #Feature Extractor
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    
            
    ##############################
        #Horovod
    ##############################

    # Initialize Horovod
    hvd.init()
    

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())
    generator = generator.to(device)
    generatorS = generatorS.to(device)
    discriminator = discriminator.to(device)
    vgg_net = vgg_net.to(device)
    l1_loss = l1_loss.to(device)
    l2_loss = l2_loss.to(device)
    VGG_loss = VGG_loss.to(device)
    cross_ent = cross_ent.to(device)
    tv_loss = tv_loss.to(device)

    ##############
    #Data Loaders
    ##############
    
    #Train loader
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = args.in_memory, transform = transform)       
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    loader = DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers, sampler = train_sampler)
         
    #Test Loader
    datasetT = mydataT(GT_path = args.GT_pathT, LR_path = args.LR_pathT, in_memory = args.in_memory, transform = transform)
    # Partition dataset among workers using DistributedSampler
    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    loaderT = DataLoader(datasetT, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    
    #Horovod Scaler
    lr_scaler = hvd.size()
    
    
    #GeneratorS Optimizer
    g_optim = optim.Adam(generatorS.parameters(), lr = 1e-4 * lr_scaler)
    g_optim = hvd.DistributedOptimizer(g_optim, named_parameters=generatorS.named_parameters(prefix='generatorS'))
    
    #Discriminator Optimizer
    d_optim = optim.Adam(discriminator.parameters(), lr = 1e-4 * lr_scaler)    
    d_optim = hvd.DistributedOptimizer(d_optim,named_parameters=discriminator.named_parameters(prefix='discriminator'))
        
    pre_epoch = 0
    fine_epoch = 0
              
    #scheduler = optim.lr_scheduler.StepLR(g_optim, step_size = 2000, gamma = 0.1)
    # set up learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(g_optim, milestones=[2000, 4000, 6000], gamma=0.1)

    
    # Horovod: broadcast parameters & optimizer state for Generator.
    hvd.broadcast_parameters(generatorS.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(g_optim, root_rank=0)

    # Horovod: broadcast parameters & optimizer state for Discriminator.
    hvd.broadcast_parameters(discriminator.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(d_optim, root_rank=0)
    
    # Horovod: broadcast parameters state for Teacher Generator.
    hvd.broadcast_parameters(generator.state_dict(), root_rank=0)
    
    best_g_loss = float('inf')
    g_losses = []
    
    best_psnr = 0.0
    while fine_epoch < args.fine_train_epoch:
        
        scheduler.step()
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)
            #print(gt.shape)
            #print(lr.shape)

            real_label = torch.ones((lr.size(0), 1)).to(device)
            fake_label = torch.zeros((lr.size(0), 1)).to(device)
                        
            ## Training Discriminator
            output, _ = generatorS(lr)
            #print(output.shape)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)
            
            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            # synchronize gradients
            g_optim.synchronize()
            
            with g_optim.skip_synchronize():
            
                ## Training Generator
                output, _ = generatorS(lr)
                fake_prob = discriminator(output)
            
                
            
                #Real image loss
                L2_loss = l2_loss(output, gt)
                L1_loss = l1_loss(output, gt)


                #Distillation loss
                outputT,_ = generator(lr)

                distill_loss1 = l1_loss(outputT, output)
            
            
            
                distill_loss2 = l2_loss(outputT, output)


                #######################
                
                # VGG and adversarial loss
                _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer = args.feat_layer)
                percep_loss = args.vgg_rescale_coeff * _percep_loss
                adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
                
                #total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat)**2)
            
                g_loss = percep_loss + adversarial_loss  + L2_loss + distill_loss1 + distill_loss2 + L1_loss
            
                g_optim.zero_grad()
                d_optim.zero_grad()
                g_loss.backward()
                g_optim.step()
            
                #synchronize gradients
                d_optim.synchronize()
            
                #Update Learning Rate
                #scheduler.step()

        #Model Evalution
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            
            for j, val_data in enumerate(loaderT):
                gt = val_data['GT'].to(device)
                lr = val_data['LR'].to(device)
                
                bs, c, h, w = lr.size()
                gt = gt[:, :, : h * args.scale, : w *args.scale]
                
                output, _ = generatorS(lr)
                
                output = output[0].cpu().numpy()
                output = np.clip(output, -1.0, 1.0)
                gt = gt[0].cpu().numpy()
                
                
                output = (output + 1.0) / 2.0
                gt = (gt + 1.0) / 2.0
                
                
                output = output.transpose(1,2,0)
                gt = gt.transpose(1,2,0)
                
                
                
                y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
                y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
                
                psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range = 1.0)
                psnr_list.append(psnr)
                
                
                ssim_val = ssim(y_output, y_gt, data_range=y_gt.max() - y_gt.min(), win_size=11,multichannel=True)
                ssim_list.append(ssim_val)

                output = Image.fromarray((output * 255.0).astype(np.uint8))
                gt = Image.fromarray((gt * 255.0).astype(np.uint8))
                lr = Image.fromarray((lr[0].cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8))
                combined_image = Image.new('RGB', (gt.width, gt.height*3))
                combined_image.paste(gt, (0, 0))
                combined_image.paste(lr, (0, gt.height))
                combined_image.paste(output, (0, gt.height*2))
                combined_filename = os.path.join(args.output_dir, f"combined_{j}.png")
                combined_image.save(combined_filename)

            
            #f.write('avg psnr : %04f' % np.mean(psnr_list))
            #f.write('Average PSNR: %f, Average SSIM: %f' % (np.mean(psnr_list), np.mean(ssim_list)))
            psnr_avg = np.mean(psnr_list)
            ssim_avg = np.mean(ssim_list)
            
            #psnr_avg = psnr_sum / len(loaderT)
            #ssim_avg = ssim_sum / len(loaderT)
            with open(args.results_file, 'a') as file:
                file.write(f"Epoch {fine_epoch}: PSNR = {psnr_avg:.4f}, SSIM = {ssim_avg:.4f}\n")
        
        
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save(generatorS.state_dict(), os.path.join(args.model_dir, f"epoch_{fine_epoch}psnr{psnr_avg:.2f}.pt"))   
            
            
            
        if fine_epoch % 2 ==0:
            torch.save(generatorS.state_dict(), 'model/SRGAN_gene_%03d.pt'%fine_epoch)
            torch.save(discriminator.state_dict(), 'model/SRGAN_discrim_%03d.pt'%fine_epoch) 
        
                    
        fine_epoch += 1
        
  



'''
def train(args):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #cuda = torch.cuda.is_available()
    transform  = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    

    ################
    # Models Loader
    ################

    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num, scale=args.scale)
    generatorS = GeneratorS(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 4, scale=args.scale)       
    discriminator = Discriminator(patch_size = args.patch_size * args.scale)
    
    if args.fine_tuning:        
        generatorS.load_state_dict(torch.load(args.generator_pathS))
        print("pre-trained student model model is loaded")
        print("path : %s"%(args.generator_pathS))
        
        discriminator.load_state_dict(torch.load(args.discriminator_path))
        print("pre-trained discriminator model is loaded")
        print("path : %s"%(args.discriminator_path))
    
    #discriminator = discriminator.to(device)
    discriminator.train()
   
    generator.load_state_dict(torch.load(args.generator_path))
    print("pre-trained Teacher model is loaded")
    print("path : %s"%(args.generator_path))
        
    #generator = generator.to(device)
    generator.eval()

    #generatorS = generatorS.to(device)
    generatorS.train()
    
    #### Train using perceptual & adversarial loss
    vgg_net = vgg19()
    vgg_net = vgg_net.eval()
    
    
    ###################
    # Loss Functions
    ###################
    
    l1_loss = torch.nn.L1Loss() 
    l2_loss = nn.MSELoss()
    VGG_loss = perceptual_loss(vgg_net) #Feature Extractor
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    
            
    ##############################
        #Horovod
    ##############################

    # Initialize Horovod
    hvd.init()
    if device:

        # Pin GPU to be used to process local rank (one GPU per process)
        torch.cuda.set_device(hvd.local_rank())
        generator = generator.cuda()
        generatorS = generatorS.cuda()
        discriminator = discriminator.cuda()
        vgg_net = vgg_net.cuda()
        l1_loss = l1_loss.cuda()
        l2_loss = l2_loss.cuda()
        VGG_loss = VGG_loss.cuda()
        cross_ent = cross_ent.cuda()
        tv_loss = tv_loss.cuda()

    ##############
    #Data Loaders
    ##############
    
    #Train loader
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = args.in_memory, transform = transform)       
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, sampler = train_sampler)
         
    #Test Loader
    datasetT = mydataT(GT_path = args.GT_pathT, LR_path = args.LR_pathT, in_memory = args.in_memory, transform = transform)
    # Partition dataset among workers using DistributedSampler
    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    loaderT = DataLoader(datasetT, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    
    #Horovod Scaler
    lr_scaler = hvd.size()
    
    
    #GeneratorS Optimizer
    g_optim = optim.Adam(generatorS.parameters(), lr = 1e-4 * lr_scaler)
    g_optim = hvd.DistributedOptimizer(g_optim, named_parameters=generatorS.named_parameters(prefix='generatorS'))
    
    #Discriminator Optimizer
    d_optim = optim.Adam(discriminator.parameters(), lr = 1e-4 * lr_scaler)    
    d_optim = hvd.DistributedOptimizer(d_optim,named_parameters=discriminator.named_parameters(prefix='discriminator'))
        
    pre_epoch = 0
    fine_epoch = 3998
              
    #scheduler = optim.lr_scheduler.StepLR(g_optim, step_size = 2000, gamma = 0.1)
    # set up learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(g_optim, milestones=[100, 150, 200], gamma=0.1)

    
    # Horovod: broadcast parameters & optimizer state for Generator.
    hvd.broadcast_parameters(generatorS.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(g_optim, root_rank=0)

    # Horovod: broadcast parameters & optimizer state for Discriminator.
    hvd.broadcast_parameters(discriminator.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(d_optim, root_rank=0)
    
    # Horovod: broadcast parameters state for Teacher Generator.
    hvd.broadcast_parameters(generator.state_dict(), root_rank=0)
    
    best_g_loss = float('inf')
    g_losses = []
    
    best_psnr = 0.0
    while fine_epoch < args.fine_train_epoch:
        
        #scheduler.step()
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)
            #print(gt.shape)
            #print(lr.shape)

            real_label = torch.ones((lr.size(0), 1)).to(device)
            fake_label = torch.zeros((lr.size(0), 1)).to(device)
                        
            ## Training Discriminator
            output, _ = generatorS(lr)
            #print(output.shape)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)
            
            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            # synchronize gradients
            g_optim.synchronize()
            
            ## Training Generator
            output, _ = generatorS(lr)
            fake_prob = discriminator(output)
            
            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer = args.feat_layer)
            
            #Real image loss
            L2_loss = l2_loss(output, gt)
            L1_loss = l1_loss(output, gt)


            #Distillation loss
            outputT,_ = generator(lr)

            distill_loss1 = l2_loss(outputT, output)
            
            
            
            distill_loss2 = l1_loss(outputT, output)


            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            #total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat)**2)
            
            g_loss = percep_loss + adversarial_loss  + L2_loss + distill_loss1 + distill_loss2 + L1_loss
            
            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            
            #synchronize gradients
            d_optim.synchronize()
            
            #Update Learning Rate
            scheduler.step()

        #Model Evalution
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            
            for j, val_data in enumerate(loaderT):
                gt = val_data['GT'].to(device)
                lr = val_data['LR'].to(device)
                
                bs, c, h, w = lr.size()
                gt = gt[:, :, : h * args.scale, : w *args.scale]
                
                output, _ = generatorS(lr)
                
                output = output[0].cpu().numpy()
                output = np.clip(output, -1.0, 1.0)
                gt = gt[0].cpu().numpy()
                
                
                output = (output + 1.0) / 2.0
                gt = (gt + 1.0) / 2.0
                
                
                output = output.transpose(1,2,0)
                gt = gt.transpose(1,2,0)
                
                
                
                y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
                y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
                
                psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range = 1.0)
                psnr_list.append(psnr)
                
                
                ssim_val = ssim(y_output, y_gt, data_range=y_gt.max() - y_gt.min(), win_size=11,multichannel=True)
                ssim_list.append(ssim_val)

                output = Image.fromarray((output * 255.0).astype(np.uint8))
                gt = Image.fromarray((gt * 255.0).astype(np.uint8))
                lr = Image.fromarray((lr[0].cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8))
                combined_image = Image.new('RGB', (gt.width, gt.height*3))
                combined_image.paste(gt, (0, 0))
                combined_image.paste(lr, (0, gt.height))
                combined_image.paste(output, (0, gt.height*2))
                combined_filename = os.path.join(args.output_dir, f"combined_{j}.png")
                combined_image.save(combined_filename)

            
            #f.write('avg psnr : %04f' % np.mean(psnr_list))
            #f.write('Average PSNR: %f, Average SSIM: %f' % (np.mean(psnr_list), np.mean(ssim_list)))
            psnr_avg = np.mean(psnr_list)
            ssim_avg = np.mean(ssim_list)
            
            #psnr_avg = psnr_sum / len(loaderT)
            #ssim_avg = ssim_sum / len(loaderT)
            with open(args.results_file, 'a') as file:
                file.write(f"Epoch {fine_epoch}: PSNR = {psnr_avg:.4f}, SSIM = {ssim_avg:.4f}\n")
        
        
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save(generatorS.state_dict(), os.path.join(args.model_dir, f"epoch_{fine_epoch}psnr{psnr_avg:.2f}.pt"))   
            
            
            
        if fine_epoch % 2 ==0:
            torch.save(generatorS.state_dict(), 'model/SRGAN_gene_%03d.pt'%fine_epoch)
            torch.save(discriminator.state_dict(), 'model/SRGAN_discrim_%03d.pt'%fine_epoch) 
        
                    
        fine_epoch += 1
        
'''  

# In[ ]:

def test(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    
    f = open('./result.txt', 'w')
    psnr_list = []
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w *args.scale]

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            gt = gt[0].cpu().numpy()

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1,2,0)
            gt = gt.transpose(1,2,0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            
            psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range = 1.0)
            psnr_list.append(psnr)
            f.write('psnr : %04f \n' % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png'%i)

        f.write('avg psnr : %04f' % np.mean(psnr_list))


def test_only(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = testOnly_data(LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            lr = te_data['LR'].to(device)
            output, _ = generator(lr)
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1,2,0)
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png'%i)



