import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from taming.models import vqgan

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from quant_scripts.brecq_quant_model import QuantModel
from quant_scripts.brecq_quant_layer import QuantModule
from quant_scripts.brecq_layer_recon import layer_reconstruction
from quant_scripts.brecq_block_recon import block_reconstruction_single_input, block_reconstruction_two_input
from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer

from tqdm import tqdm
import copy


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
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")  
    model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")
    return model

if __name__ == '__main__':
    num_calibration_batches = 1

    data_path = 'imagenet_samples_ddim_50steps_sd.pth'
    data_list = torch.load(data_path, map_location='cpu')
    print('load data: ', data_path)
    model = get_model()
    print("get_model")
    
