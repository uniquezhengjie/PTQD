## To collect input data, remember to uncomment line 987-988 in ldm/models/diffusion/ddpm.py and comment them after finish collecting.
import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")  
    model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")
    return model

if __name__ == '__main__':
    model = get_model()
    sampler = DDIMSampler(model)


    # classes = [25, 187, 448, 992]   # define classes to be sampled here
    classes = [i for i in range(1000)]
    n_samples_per_class = 1

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0   

    datasets = []
    for c in classes:
        name = label_to_name(c).split(',')[0]
        datasets.append("This is a photo of a {}".format(name))

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(n_samples_per_class * [""])
            
            for class_label in classes:
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

                # x_samples_ddim = model.decode_first_stage(samples_ddim)
                # x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                #                             min=0.0, max=1.0)
                # all_samples.append(x_samples_ddim)
                all_samples.append(samples_ddim)

    ## save all_samples
    # 为了提供给decoder模型做量化
    torch.save(all_samples, 'imagenet_samples_ddim_{}steps_sd.pth'.format(ddim_steps))
    # sys.exit(0)
    ## save diffusion input data
    import ldm.globalvar as globalvar   
    input_list = globalvar.getInputList()
    torch.save(input_list, 'imagenet_input_{}steps_sd.pth'.format(ddim_steps))
    sys.exit(0)