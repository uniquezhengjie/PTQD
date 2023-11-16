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
from torch import autocast
import time

def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

if __name__ == '__main__':
    config = "optimizedSD/v1-inference.yaml"
    sd = load_model_from_config("models/ldm/stable-diffusion-v1/model.ckpt")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)
    
    config = OmegaConf.load(config)

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = 1
    model.cdevice = "cuda"
    model.turbo = False

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = "cuda"

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd
    
    model.half()
    modelCS.half()
    precision_scope = autocast

    classes = [25, 187]   # define classes to be sampled here
    # n_samples_per_class = 6
    n_samples_per_class = 1

    ddim_steps = 50
    ddim_eta = 0.0
    scale = 7.5
    
    # ddim_steps = 20
    # ddim_eta = 0.0
    # scale = 7.5

    datasets = []
    for c in classes:
        name = label_to_name(c).split(',')[0]
        datasets.append("This is a photo of a {}".format(name))

    all_samples = list()

    with torch.no_grad():
        with precision_scope("cuda"):
            modelCS.to("cuda")
            uc = modelCS.get_learned_conditioning(n_samples_per_class * [""])
            
            for class_label in datasets:
                modelCS.to("cuda")
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                c = modelCS.get_learned_conditioning(n_samples_per_class*[class_label])
                mem = torch.cuda.memory_allocated() / 1e6
                modelCS.to("cpu")
                while torch.cuda.memory_allocated() / 1e6 >= mem:
                    time.sleep(1)
                    
                samples_ddim = model.sample(S=ddim_steps,
                                                conditioning=c,
                                                shape=[n_samples_per_class,4, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)
                
                modelFS.to("cuda")
                x_samples_ddim = modelFS.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                mem = torch.cuda.memory_allocated() / 1e6
                modelFS.to("cpu")
                while torch.cuda.memory_allocated() / 1e6 >= mem:
                    time.sleep(1)
                    
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

