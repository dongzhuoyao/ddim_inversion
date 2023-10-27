"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import kornia
import numpy as np
import torch as th
torch = th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

class Retinex_shading(nn.Module):
    def __init__(self):
        super(Retinex_shading, self).__init__()
    
    def get_gauss_kernel(self, sigma, dim=2):
        # calculate kernel size
        ksize = int(np.floor(sigma*6)/2)*2+1 
        # create 1D kernel
        k_1D = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        k_1D = torch.exp(-k_1D**2 / (2 * sigma**2))
        k_1D /= k_1D.sum()
        # if 1D, return
        if dim == 1:
            return k_1D
        # if 2D, compute the 2D kernel
        elif dim == 2:
            return k_1D[:, None].mm(k_1D[None, :])

    def gauss_blur_original(self, img, sigma):
        kernel = self.get_gauss_kernel(sigma, 1)
        kernel = kernel.float().view(1, 1, -1, 1).cuda()
        padding_size = kernel.shape[2] // 2
        t = F.conv2d(img, kernel.repeat(img.shape[1], 1, 1, 1), stride=1, padding=(padding_size, 0), groups=img.shape[1])
        kernel_t = kernel.transpose(2, 3)
        t = F.conv2d(t, kernel_t.repeat(img.shape[1], 1, 1, 1), stride=1, padding=(0, padding_size), groups=img.shape[1])
        return t
    
    def gauss_blur_recursive(self, img, sigma):
        # Implement the recursive filter here
        pass

    def gauss_blur(self, img, sigma, method='original'):
        if method == 'original':
            return self.gauss_blur_original(img, sigma)
        elif method == 'recursive':
            return self.gauss_blur_recursive(img, sigma)

    def retinex_MSR(self, img, sigmas=[15, 80, 250], weights=None):
        if weights is None:
            weights = torch.ones(len(sigmas)) / len(sigmas)
        else:
            weights = torch.tensor(weights)
            if not torch.abs(weights.sum() - 1) < 0.00001:
                raise ValueError('Sum of weights must be 1!')
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        batch, channels, height, width = img.shape
        ret = torch.zeros_like(img)

        for i in range(channels):
            channel = img[:, i, :, :].float()
            r = torch.zeros_like(channel)

            for k, sigma in enumerate(sigmas):
                blurred_channel = self.gauss_blur(channel.unsqueeze(1), sigma)
                r += (torch.log(blurred_channel.squeeze(1) + 1)) * weights[k]

            mmin = torch.min(r)
            mmax = torch.max(r)
            stretch = (r - mmin) / (mmax - mmin)
            ret[:, i, :, :] = stretch

        return ret



class gaussRetinexCal(Retinex_shading):
    """
    Class to calculate the cross ratio, using a discrete filter.
    """

    def __init__(self):
        super(gaussRetinexCal, self).__init__()
    def forward(self, img):
        Retinex = self.retinex_MSR(img)
        return Retinex




def gaussian_2d(x, y, sigma_x=10, sigma_y=10, mu_x=32, mu_y=32):
    return torch.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))


def gaussian_mask(size, sigma_x, sigma_y, center=None):
    """Generate a 2D Gaussian mask.
    
    Parameters:
        size (tuple): The size of the mask (height, width).
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        center (tuple, optional): (x, y) position of the center. 
                                Defaults to the center of the mask.
                                
    Returns:
        torch.Tensor: The generated Gaussian mask.
    """
    if center is None:
        center = (size[1] // 2, size[0] // 2)
    
    y, x = torch.meshgrid([torch.arange(0, size[0]), torch.arange(0, size[1])])
    y = y.float() - center[1]
    x = x.float() - center[0]
    
    mask = torch.exp(-(x*x / (2*sigma_x**2) + y*y / (2*sigma_y**2)))
    # mask = mask / mask.sum()  # Normalize the mask to make the sum of all values equal to 1
    
    return mask



def SGaussian(center = [0,0],sigma_x = 300,sigma_y = 300):
    size = (256, 256)  # size of the mask
    sigma_x = sigma_x  # standard deviation in the x-axis
    sigma_y = sigma_y   # standard deviation in the y-axis
    
    mask = gaussian_mask(size, sigma_x, sigma_y,center = center)
    mask = mask.unsqueeze(0).cuda()
    return mask.repeat(3,1,1)


def get_non_zero_mask(inputs):
    mask = torch.zeros_like(inputs)
    input_1d = torch.sum(inputs,dim=1)
    # print(input_1d.shape)
    mask_idx = torch.nonzero(input_1d)
    mask[mask_idx] = 1
    return mask

def main():
    import random
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    args = create_argparser().parse_args()
    sample_dir = os.getcwd()+ '/examples/retinex'
    os.makedirs(sample_dir, exist_ok=True)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
   
    # # load real image as target 
    # image_path = 'target.png'  # 替换为你的图片路径
    # image = Image.open(image_path)


    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])
    # tensor_image = transform(image)
    # targets = tensor_image.unsqueeze(0).repeat(args.batch_size,1,1,1).cuda()
    # get_retinex_1 = gaussRetinexCal().cuda()
    # targets = get_retinex_1(targets)
    targets = SGaussian(center=[0,128],sigma_x=300,sigma_y=400).unsqueeze(0).repeat(args.batch_size,1,1,1)
    targets_2 = 1 - SGaussian(center=[128,0],sigma_x=400,sigma_y=200).unsqueeze(0).repeat(args.batch_size,1,1,1)
    # targets = get_retinex()
    # target_platte = th.zeros((3,256,256)).cuda()
    # target_platte[:,:128,:128] = 0.2
    # target_platte[:,128:,128:] = 1
    # target_platte[:,128:,:128] = 0.2
    # target_platte[:,:128,128:] = 1
    # targets = target_platte.unsqueeze(0).repeat(args.batch_size,1,1,1)

    def cond_fn(x, t, y=None, **kwargs):
        if t[0] < 500 and t[0]>200:
            print('t',t[0])
            with torch.enable_grad():
                x_in = (x+0).float().requires_grad_(True)
                get_retinex = gaussRetinexCal().cuda()
                x_in = get_retinex(x_in)
                # x_in = x.detach().requires_grad_(True)
                # loss = torch.nn.MSELoss(reduction='none')
                print(x_in.shape)
                print(targets.shape)
                x_in_g = x_in.mean(dim=1,keepdim=True)
                ## fake 
                # x_fake = x_in_g[1,...]
                # x_fake = x_fake.unsqueeze(0).repeat(x_in.shape[0],1,1,1)
                # targets = targets.repeat(x_in.shape[0],3,1,1)
                # logits = histo_discriminator(targets,input_x)
                # pr = loss(x_in,targets)  
                # pr = torch.abs(x_in_g-x_fake)
                pr = torch.abs(x_in_g-targets.mean(dim=1,keepdim=True))
                #non gray scale
                # pr = torch.abs(x_in-targets)
                # pr = discriminator(x_in, t[:int(t.shape[0] / 2)] / 999, condition=y[:int(t.shape[0] / 2)])
                # mask = get_non_zero_mask(pr)
                pr = pr
                print(pr.mean())
                pr = torch.clip(pr, min=1e-5, max=1 - 1e-5)

                log_density_ratio = torch.log(pr) - torch.log(1 - pr)
                save_image(x_in, sample_dir + f"/test_0.png", nrow=int(np.sqrt(x_in.shape[0])), normalize=True, value_range=(-1, 1))
                save_image(targets, sample_dir + f"/test_target.png", nrow=int(np.sqrt(x_in.shape[0])), normalize=True, value_range=(-1, 1))
                dg = torch.autograd.grad(log_density_ratio.sum(), x_in_g,retain_graph=False)[0]
                dg = torch.nn.functional.normalize(dg,dim=1)
                dg = dg.clamp(-1+1e-5,1. - 1e-5)

            # dg = torch.cat([dg, dg], dim=0)
            return dg*0.2
            
        else:
            return torch.zeros_like(x)


    def cond_fn_2(x, t, y=None, **kwargs):
        if t[0] < 500 and t[0]>200:
            print('t',t[0])
            with torch.enable_grad():
                x_in = (x+0).float().requires_grad_(True)
                get_retinex = gaussRetinexCal().cuda()
                x_in = get_retinex(x_in)
                # x_in = x.detach().requires_grad_(True)
                # loss = torch.nn.MSELoss(reduction='none')
                print(x_in.shape)
                print(targets_2.shape)
                x_in_g = x_in.mean(dim=1,keepdim=True)
                x_in_g = x_in_g.clamp(1e-5,1-1e-5)
                ## fake 
                # x_fake = x_in_g[1,...]
                # x_fake = x_fake.unsqueeze(0).repeat(x_in.shape[0],1,1,1)
                # targets = targets.repeat(x_in.shape[0],3,1,1)
                # logits = histo_discriminator(targets,input_x)
                # pr = loss(x_in,targets)  
                # pr = torch.abs(x_in_g-x_fake)
                pr = torch.abs(x_in_g-targets_2.mean(dim=1,keepdim=True))
                print(pr.mean())
                #non gray scale
                # pr = torch.abs(x_in-targets)
                # pr = discriminator(x_in, t[:int(t.shape[0] / 2)] / 999, condition=y[:int(t.shape[0] / 2)])
                # mask = get_non_zero_mask(pr)
                pr = pr
                pr = torch.clip(pr, min=1e-5, max=1 - 1e-5)

                log_density_ratio = torch.log(pr) - torch.log(1 - pr)
                save_image(x_in, sample_dir + f"/test_0.png", nrow=int(np.sqrt(x_in.shape[0])), normalize=True, value_range=(-1, 1))
                save_image(targets, sample_dir + f"/test_target.png", nrow=int(np.sqrt(x_in.shape[0])), normalize=True, value_range=(-1, 1))
                dg = torch.autograd.grad(log_density_ratio.sum(), x_in_g,retain_graph=False)[0]
                dg = torch.nn.functional.normalize(dg,dim=1)
                dg = dg.clamp(-1+1e-5,1. - 1e-5)

            # dg = torch.cat([dg, dg], dim=0)
            return dg*0.2
            
        else:
            return torch.zeros_like(x)


    logger.log("sampling...")
    all_images = []
    all_labels = []
    count = 0
    while len(all_images) * args.batch_size < args.num_samples:
        y = torch.randint(low=0, high=1000, size=(args.batch_size,), device=device)
        n = y.shape[0]
        z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
        # Setup classifier-free guidance:
        # z = torch.cat([z, z], 0)
        model_kwargs = dict(y=y)
        sample_fn = (
            diffusion.ddim_sample_loop
        )
        samples = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),z,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,
            eta=0.0
        )

        if True:
            # taohu
            sample = (samples + 1) * 0.5
            save_image(sample, "sample.png")
            sample = (sample - 0.5) * 2.0
            noise_z = diffusion.ddim_sample_reverse_loop(
                model,
                shape=(args.batch_size, 3, args.image_size, args.image_size),
                noise=sample,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=dist_util.dev(),
                eta=0.0,
            )
            print(noise_z.shape)
            print(noise_z.min(), noise_z.max(), noise_z.mean(), noise_z.std())
            recovered_sample = diffusion.ddim_sample_loop(
                model,
                shape=(args.batch_size, 3, args.image_size, args.image_size),
                noise=noise_z,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=dist_util.dev(),
                eta=0.0,
            )
            recovered_sample = (recovered_sample + 1) * 0.5
            save_image(recovered_sample, "recovered_sample.png")

            exit(0)

        r = np.random.randint(1000000)+1
        if count == 0:
            save_image(samples, sample_dir + f"/sample_{r}_bed_0_0.png", nrow=int(np.sqrt(samples.shape[0])), normalize=True, value_range=(-1, 1))
        sample = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8)
        
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=64,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
