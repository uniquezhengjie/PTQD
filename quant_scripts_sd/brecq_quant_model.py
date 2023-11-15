import torch.nn as nn
# from quantization.brecq_quant_block import specials, BaseQuantBlock
from quant_scripts.brecq_quant_layer import QuantModule, StraightThrough
from quant_scripts.brecq_fold_bn import search_fold_and_remove_bn


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        # search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) and 'skip' not in name and 'op' not in name:  ## keep skip connection full-precision
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, image,t,context):
        return self.model(image,t,context)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        # module_list[1].weight_quantizer.bitwidth_refactor(8)
        # module_list[1].act_quantizer.bitwidth_refactor(8)
        module_list[2].weight_quantizer.bitwidth_refactor(8) ## it is a input layer
        module_list[2].act_quantizer.bitwidth_refactor(8)

        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].act_quantizer.bitwidth_refactor(8)

        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True
        # module_list[1].ignore_reconstruction = True
        module_list[2].ignore_reconstruction = True
        module_list[-1].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True


