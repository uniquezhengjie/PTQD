import os
import sys
import time
import numpy as np

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

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
_ = torch.manual_seed(191009)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.conv_pre = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.hard_swish = torch.nn.Hardswish()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()
        self.mul = nn.quantized.FloatFunctional()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        
        x = self.conv_pre(x)
        # x = self.quant(x)
        # h = x
        x = self.conv(x)
        x = self.relu(x)
        # x = nonlinearity(x)
        # x = self.mul.mul(x,self.sigmoid(x))
        # x = self.swish(x)
        # x = self.hard_swish(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        # x = h+x
        # x = torch.softmax(x,0)
        # x = torch.nn.functional.softmax(x)
        # x = x*torch.nn.functional.sigmoid(x)
        # x = self.softmax(x)
        # x = self.skip_add.add(h, x)
        # a = torch.randn(1,100,10).to('cpu')
        # b = torch.randn(1,10,100).to('cpu')
        # x = torch.bmm(a,b)
        # x = self.dequant(x)
        return x
    
# create a model instance
model_fp32 = M()
model_fp32.to('cpu')
# model must be set to eval mode for static quantization logic to work
model_fp32.eval()
example_inputs = torch.randn(4, 1, 4, 4).to('cpu')
print(model_fp32(example_inputs))

qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)


prepared_model = prepare_fx(model_fp32, qconfig_mapping, example_inputs)
# print(prepared_model.graph)

prepared_model.eval()
with torch.no_grad():
    prepared_model(example_inputs)

quantized_model = convert_fx(prepared_model)
# print(quantized_model)
print(quantized_model(example_inputs))
