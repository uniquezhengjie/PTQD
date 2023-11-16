import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from taming.models import vqgan

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from imagenet_2012_labels import label_to_name

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if __name__ == '__main__':
    model = get_model()
    sampler = DDIMSampler(model)

    classes = [25, 187, 448, 992]   # define classes to be sampled here
    n_samples_per_class = 6
    # n_samples_per_class = 1

    ddim_steps = 50
    ddim_eta = 0.0
    scale = 7.5

    datasets = []
    for c in classes:
        name = label_to_name(c).split(',')[0]
        datasets.append("This is a photo of a {}".format(name))

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(n_samples_per_class * [""])
            
            for class_label in datasets:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                c = model.get_learned_conditioning(n_samples_per_class*[class_label])
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[4, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)
                print('finish {} class'.format(class_label))

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    image_to_save.save("imagenet_result_step{}_eta{}_scale{}_sd.jpg".format(ddim_steps, ddim_eta, scale))

