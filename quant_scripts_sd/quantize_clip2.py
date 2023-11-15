from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
import requests
from io import BytesIO
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import torch
from datasets import load_dataset
import logging
import nncf
from openvino.runtime import Core, serialize
import numpy as np
from scipy.special import softmax
from openvino.runtime import compile_model

import sys,os
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan

import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn


def check_text_data(data):
    """
    Check if the given data is text-based.
    """
    if isinstance(data, str):
        return True
    if isinstance(data, list):
        return all(isinstance(x, str) for x in data)
    return False

def collate_fn(example, text_column="caption", max_length=77):
    """
    Preprocesses an example by loading and transforming image and text data.
    Checks if the text data in the example is valid by calling the `check_text_data` function.
    Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
    If there is any error during the download process, returns None.
    Returns the preprocessed inputs with transformed image and text data.
    """
    assert len(example) == 1
    example = example[0]

    if not check_text_data(example[text_column]):
        raise ValueError("Text data is not valid")

    inputs = processor(text=example[text_column], truncation=True, max_length=max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

    if inputs['input_ids'].shape[1] > max_length:
        return None
    return inputs

def prepare_calibration_data(dataloader, init_steps):
    """
    This function prepares calibration data from a dataloader for a specified number of initialization steps.
    It iterates over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    print(f"Fetching {init_steps} for the initialization...")
    counter = 0
    for batch in dataloader:
        print(counter)
        if counter == init_steps:
            break
        if batch:
            counter += 1
            with torch.no_grad():
                data.append(batch["input_ids"].to("cpu"))
    return data


def prepare_dataset(opt_init_steps=300, max_train_samples=1000):
    """
    Prepares a vision-text dataset for quantization.
    """
    dataset = load_dataset("conceptual_captions", streaming=True)
    train_dataset = dataset["train"].shuffle(seed=42, buffer_size=max_train_samples)
    dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1)
    calibration_data = prepare_calibration_data(dataloader, opt_init_steps)
    return calibration_data

def prepare_dataset2(inputs):
    data = []
    data.append(inputs["input_ids"].to("cpu"))
    return data

# model_sd = get_model()
# model = model_sd.cond_stage_model
# max_length = 77

# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
# max_length = model.config.text_config.max_position_embeddings
# processor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# fp16_model_path = 'clip-vit-base-patch16.xml'
# int8_model_path = 'clip-vit-base-patch16_int8.xml'

model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
max_length = model.config.max_position_embeddings
processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

fp16_model_path = 'clip-vit-large-patch14_txt.xml'
int8_model_path = 'clip-vit-large-patch14_txt_int8.xml'

core = Core()

nncf.set_log_level(logging.ERROR)

calibration_data = prepare_dataset()
calibration_dataset = nncf.Dataset(calibration_data)

# input_labels = ['cat']
# text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

# inputs = processor(text_descriptions, truncation=True, max_length=max_length, return_length=True,
#                                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
# calibration_data = prepare_dataset2(inputs)

ov_model = core.read_model(fp16_model_path)

if len(calibration_data) == 0:
    raise RuntimeError(
        'Calibration dataset is empty. Please check internet connection and try to download images manually.'
    )

calibration_dataset = nncf.Dataset(calibration_data)
quantized_model = nncf.quantize(
    model=ov_model,
    calibration_dataset=calibration_dataset,
    model_type=nncf.ModelType.TRANSFORMER,
)

serialize(quantized_model, int8_model_path)


# input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
# text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

# inputs = processor(text_descriptions, truncation=True, max_length=max_length, return_length=True,
#                                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
# compiled_model = compile_model(int8_model_path)
# logits_per_image_out = compiled_model.output(0)
# ov_logits_per_image = compiled_model(dict(inputs))[logits_per_image_out]
# probs = softmax(ov_logits_per_image, axis=1)
# visualize_result(image, input_labels, probs[0])



