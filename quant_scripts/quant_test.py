import torch
from torch import nn

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class NonLinearity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        return x*self.sigmoid(x)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.conv_pre = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.swish = NonLinearity()
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
        x = self.quant(x)
        h = x
        x = self.conv(x)
        # x = self.relu(x)
        # x = nonlinearity(x)
        x = self.mul.mul(x,self.sigmoid(x))
        # x = self.swish(x)
        # x = self.hard_swish(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        # x = h+x
        # x = self.softmax(x)
        x = self.skip_add.add(h, x)
        x = self.dequant(x)
        return x
    
# create a model instance
model_fp32 = M()
model_fp32.to('cpu')
# model must be set to eval mode for static quantization logic to work
model_fp32.eval()
input_fp32 = torch.randn(4, 1, 4, 4).to('cpu')
print(model_fp32(input_fp32))

model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_fp32.conv_pre.qconfig = None

model_fp32_prepared = torch.ao.quantization.prepare(model_fp32, inplace=True)

model_fp32_prepared(input_fp32)

model_int8 = torch.ao.quantization.convert(model_fp32_prepared, inplace=True)
res = model_int8(input_fp32)
print(res)