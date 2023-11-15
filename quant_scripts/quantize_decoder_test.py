import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan

import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import DDPM

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from torch.ao.quantization import get_default_qconfig, QConfigMapping,get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # model.cuda()
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if __name__ == '__main__':
    num_calibration_batches = 10
    # data_path = 'imagenet_samples_ddim_20steps.pth'
    data_path = 'generated/imagenet_samples_ddim_100steps_1.0eta.pth'
    data_list = torch.load(data_path, map_location='cpu')

    # fx_graph_mode_model_file_path = 'quantized_decoder/decoder_fx_graph_mode_quantized.pth'
    fx_graph_mode_model_file_path = 'quantized_decoder/decoder_fx_graph_mode_quantized_100step.pth'
    quantized_modelFS = torch.jit.load(fx_graph_mode_model_file_path)

    # model.first_stage_model = quantized_modelFS

    # x_samples_ddim = model.decode_first_stage(torch.from_numpy(np.zeros([1, 3, 64, 64],dtype=np.float32)))
    x_samples_ddim = quantized_modelFS(data_list[1])
    # x_samples_ddim_fp32 = modelFS(data_list[0])
    print(x_samples_ddim.min(), x_samples_ddim.max())
    
    x_samples_ddim = torch.clamp((x_samples_ddim[0]+1.0)/2.0, 
                                min=0.0, max=1.0)
    
    img = 255. * rearrange(x_samples_ddim, 'c h w -> h w c').cpu().numpy()
    print(img.astype(np.uint8).min(),img.astype(np.uint8).max())
    image_to_save = Image.fromarray(img.astype(np.uint8))
    image_to_save.save("quant_decoder2.jpg")
    
