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
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if __name__ == '__main__':
    num_calibration_batches = 1
    # config = "tiny_optimizedSD/v1-inference.yaml"
    # ckpt = "/home/reexen/projects/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
    # config = OmegaConf.load(f"{config}")
    # sd = load_model_from_config(f"{ckpt}")
    # modelFS = instantiate_from_config(config.model.params.)
    # _, _ = modelFS.load_state_dict(sd, strict=False)
    # data_path = 'imagenet_samples_ddim_20steps.pth'
    data_path = 'generated/imagenet_samples_ddim_100steps_1.0eta.pth'

    data_list = torch.load(data_path, map_location='cpu')

    model = get_model()
    modelFS = model.first_stage_model
    modelFS.to('cpu')
    modelFS.eval()

    # x_samples_ddim_fp32 = modelFS(data_list[0])
    
    # x_samples_ddim = torch.clamp((x_samples_ddim_fp32[0]+1.0)/2.0, 
    #                             min=0.0, max=1.0)
    
    # img = 255. * rearrange(x_samples_ddim, 'c h w -> h w c').cpu().numpy()
    # print(img.astype(np.uint8).min(),img.astype(np.uint8).max())
    # image_to_save = Image.fromarray(img.astype(np.uint8))
    # image_to_save.save("quant_decoder1.jpg")
    # exit()

    # # Fuse Conv, bn and relu
    # modelFS.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    # myModel.qconfig = torch.ao.quantization.default_qconfig
    # modelFS.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # modelFS.quantize.qconfig = None
    # example_inputs = torch.randn(1, 3, 64, 64).to('cpu')
    # print(model.decode_first_stage(torch.from_numpy(np.zeros([1, 3, 64, 64],dtype=np.float32))))
    # print(modelFS(data_list[0]))
    # print(modelFS)
    qconfig = get_default_qconfig("onednn")  # reduce_range=False onednn and qnnpack
    # qconfig = get_default_qconfig("x86")
    # qconfig = torch.ao.quantization.float16_static_qconfig
    # qconfig = torch.ao.quantization.default_qconfig
    qconfig_mapping = QConfigMapping().set_global(qconfig) \
                        # .set_object_type(torch.nn.Conv2d, torch.ao.quantization.float16_static_qconfig) \
                        # .set_module_name("quantize", qconfig2)
    # torch.ao.quantization.prepare(modelFS, inplace=True)
    prepared_modelFS = prepare_fx(modelFS, qconfig_mapping, data_list[0])
    
    
    # model.first_stage_model = prepared_modelFS
    # Calibrate first


    # x_samples_ddim = prepared_modelFS(example_inputs)
    print('Post Training Quantization Prepare: Inserting Observers')
    prepared_modelFS.eval()
    with torch.no_grad():
        for samples_ddim in data_list[:num_calibration_batches]:
            # samples_ddim = torch.from_numpy(np.zeros([1, 3, 64, 64],dtype=np.float32))
            # x_samples_ddim = model.decode_first_stage(samples_ddim)
            prepared_modelFS(samples_ddim)
    # print('\n Inverted Residual Block:After observer insertion \n\n', modelFS.features[1].conv)

    # # Calibrate with the training set
    # evaluate(modelFS, criterion, data_loader, neval_batches=num_calibration_batches)
    # print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    # torch.ao.quantization.convert(modelFS, inplace=True)
    quantized_modelFS = convert_fx(prepared_modelFS)
    print('Post Training Quantization: Convert done')
    # print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',modelFS.features[1].conv)
    # quantized_modelFS.to_folder("quantized_decoder","dtest")
    # torch.save(quantized_modelFS.state_dict(), "quantized_decoder.p")
    # torch.jit.save(torch.jit.script(quantized_modelFS), 'quantized_decoder/decoder_fx_graph_mode_quantized.pth')
    torch.jit.save(torch.jit.script(quantized_modelFS), 'quantized_decoder/decoder_fx_graph_mode_quantized_100step.pth')
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
    image_to_save.save("quant_decoder1.jpg")
    
