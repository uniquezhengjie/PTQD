import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from taming.models import vqgan

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddpm import DDPM

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from torch.ao.quantization import get_default_qconfig, QConfigMapping,get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

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
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")  
    model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")
    return model

if __name__ == '__main__':
    num_calibration_batches = 1000

    data_path = 'imagenet_samples_ddim_50steps_sd.pth'
    data_list = torch.load(data_path, map_location='cpu')
    print('load data: ', data_path)
    model = get_model()
    print("get_model")
    modelFS = model.first_stage_model
    modelFS.to('cpu')
    modelFS.eval()
    print('modelFS')
    qconfig = get_default_qconfig("onednn")  # reduce_range=False onednn and qnnpack
    # qconfig = get_default_qconfig("x86")
    # qconfig = torch.ao.quantization.float16_static_qconfig
    # qconfig = torch.ao.quantization.default_qconfig
    qconfig_mapping = QConfigMapping().set_global(qconfig) \
                        # .set_object_type(torch.nn.Conv2d, torch.ao.quantization.float16_static_qconfig) \
                        # .set_module_name("quantize", qconfig2)
    # torch.ao.quantization.prepare(modelFS, inplace=True)
    prepared_modelFS = prepare_fx(modelFS, qconfig_mapping, data_list[0])
    
    print('Post Training Quantization Prepare: Inserting Observers')
    prepared_modelFS.eval()
    with torch.no_grad():
        for samples_ddim in data_list[:num_calibration_batches]:
            # samples_ddim = torch.from_numpy(np.zeros([1, 3, 64, 64],dtype=np.float32))
            # x_samples_ddim = model.decode_first_stage(samples_ddim)
            prepared_modelFS(samples_ddim)

    quantized_modelFS = convert_fx(prepared_modelFS)
    print('Post Training Quantization: Convert done')

    torch.jit.save(torch.jit.script(quantized_modelFS), 'quantized_decoder/decoder_fx_graph_mode_quantized_sd.pth')
    print("Size of model after quantization")
    print_size_of_model(quantized_modelFS)
    # model.first_stage_model = quantized_modelFS

    # x_samples_ddim = model.decode_first_stage(torch.from_numpy(np.zeros([1, 3, 64, 64],dtype=np.float32)))
    x_samples_ddim = quantized_modelFS(data_list[0])
    x_samples_ddim_fp32 = modelFS(data_list[0])
    print(x_samples_ddim.min(), x_samples_ddim.max())
    
    x_samples_ddim = torch.clamp((x_samples_ddim[0]+1.0)/2.0, 
                                min=0.0, max=1.0)
    
    img = 255. * rearrange(x_samples_ddim, 'c h w -> h w c').cpu().numpy()
    print(img.astype(np.uint8).min(),img.astype(np.uint8).max())
    image_to_save = Image.fromarray(img.astype(np.uint8))
    image_to_save.save("quant_decoder1_sd.jpg")
    
