from transformers import CLIPProcessor, CLIPModel
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
from tqdm import tqdm

save_count = 0

def check_text_data(data):
    """
    Check if the given data is text-based.
    """
    if isinstance(data, str):
        return True
    if isinstance(data, list):
        return all(isinstance(x, str) for x in data)
    return False

def get_pil_from_url(url):
    """
    Downloads and converts an image from a URL to a PIL Image object.
    """
    response = requests.get(url, verify=False, timeout=20)
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")

def collate_fn(example, image_column="image_url", text_column="caption",save_to_disk=True):
    """
    Preprocesses an example by loading and transforming image and text data.
    Checks if the text data in the example is valid by calling the `check_text_data` function.
    Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
    If there is any error during the download process, returns None.
    Returns the preprocessed inputs with transformed image and text data.
    """
    assert len(example) == 1
    example = example[0]
    global save_count

    if not check_text_data(example[text_column]):
        raise ValueError("Text data is not valid")

    url = example[image_column]
    save_path = '/data2/datasets/conceptual_captions/validation/images/{}.jpg'.format(save_count)
    try:
        if os.path.exists(save_path):
            image = Image.open(save_path)
        else:
            image = get_pil_from_url(url)
        if image.size[0] < 50 or image.size[1] < 50:
            print(example[text_column],image.size)
            return None
    except Exception:
        return None
    
    if save_to_disk:
        if not os.path.exists(save_path):
            image.save(save_path)
        save_count += 1

    inputs = processor(text=example[text_column], images=[image], return_tensors="pt", padding="max_length")
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
                data.append(
                    {
                        "pixel_values": batch["pixel_values"].to("cpu"),
                        "input_ids": batch["input_ids"].to("cpu"),
                        "attention_mask": batch["attention_mask"].to("cpu")
                    }
                )
    return data

def prepare_calibration_data_val(dataloader, init_steps,device='cuda'):
    """
    This function prepares calibration data from a dataloader for a specified number of initialization steps.
    It iterates over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    pixel_values = []
    input_ids = []
    attention_mask = []
    print(f"Fetching {init_steps} for the initialization...")
    counter = 0
    for batch in dataloader:
        print(counter)
        if counter == init_steps:
            break
        if batch:
            counter += 1
            with torch.no_grad():
                pixel_values.append(batch["pixel_values"].to(device))
                input_ids.append(batch["input_ids"].to(device))
                attention_mask.append(batch["attention_mask"].to(device))
        if counter % 10 == 0:
            data.append(
                {
                        "pixel_values": torch.concat(pixel_values),
                        "input_ids": torch.concat(input_ids),
                        "attention_mask": torch.concat(attention_mask)
                }
            )
            pixel_values = []
            input_ids = []
            attention_mask = []
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

def prepare_dataset_val(opt_init_steps=1000, device='cuda'):
    """
    Prepares a vision-text dataset for quantization.
    """
    dataset = load_dataset("conceptual_captions", streaming=True)
    dataloader = torch.utils.data.DataLoader(dataset["validation"], collate_fn=collate_fn, batch_size=1)
    calibration_data = prepare_calibration_data_val(dataloader, opt_init_steps,device=device)
    return calibration_data

def prepare_dataset2(inputs):
    data = []
    data.append(
    {
        "pixel_values": inputs["pixel_values"].to("cpu"),
        "input_ids": inputs["input_ids"].to("cpu"),
        "attention_mask": inputs["attention_mask"].to("cpu"),
    }
    )
    return data

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
max_length = model.config.text_config.max_position_embeddings
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

fp16_model_path = 'clip-vit-base-patch16.xml'
int8_model_path = 'clip-vit-base-patch16_int8.xml'

# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# max_length = model.config.text_config.max_position_embeddings
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# fp16_model_path = 'clip-vit-large-patch14.xml'
# int8_model_path = 'clip-vit-large-patch14_int8.xml'

core = Core()

nncf.set_log_level(logging.ERROR)

ov_model = core.read_model(fp16_model_path)


if False:
    calibration_data = prepare_dataset()
    calibration_dataset = nncf.Dataset(calibration_data)
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
    )

    serialize(quantized_model, int8_model_path)


if True:
    model.cuda()
    model.eval()
    calibration_data = prepare_dataset_val()
    text_embeds = []
    image_embeds = []
    for input_data in tqdm(calibration_data):
        torch.cuda.empty_cache()
        with torch.no_grad():
            results = model(**input_data)
            text_embeds.append(results['text_embeds'].cpu())
            image_embeds.append(results['image_embeds'].cpu())

    # print(text_embeds)
    with torch.no_grad():
        fp32_text_embeds = torch.concat(text_embeds)
        fp32_image_embeds = torch.concat(image_embeds)
        logit_scale = model.logit_scale.exp().cpu()
        logits_per_text = torch.matmul(fp32_text_embeds, fp32_image_embeds.t()) * logit_scale
        fp32_loss = clip_loss(logits_per_text)


if True:
    calibration_data = prepare_dataset_val(device='cpu')
    compiled_model = compile_model(int8_model_path)
    # logits_per_text = compiled_model.output(1)
    text_embed_out = compiled_model.output(2)
    image_embed_out = compiled_model.output(3)

    text_embeds = []
    image_embeds = []
    for input_data in tqdm(calibration_data):
        text_embed = compiled_model(input_data)[text_embed_out]
        image_embed = compiled_model(input_data)[image_embed_out]
        text_embeds.append(torch.from_numpy(text_embed))
        image_embeds.append(torch.from_numpy(image_embed))

    with torch.no_grad():
        int8_text_embeds = torch.concat(text_embeds)
        int8_image_embeds = torch.concat(image_embeds)
        logit_scale = model.logit_scale.exp().cpu()
        logits_per_text = torch.matmul(int8_text_embeds, int8_image_embeds.t()) * logit_scale
        int8_loss = clip_loss(logits_per_text)
        # 14.3068

print(int8_text_embeds.shape)
print("fp32_loss: ",fp32_loss)
print("int8_loss: ",int8_loss)

print("L1 distance between fp32_text_embeds and int8_text_embeds", torch.dist(fp32_text_embeds,int8_text_embeds,p=1)/len(fp32_text_embeds))
print("L1 distance between fp32_image_embeds and int8_text_embeds", torch.dist(fp32_image_embeds,int8_text_embeds,p=1)/len(fp32_text_embeds))
print("L1 distance between fp32_image_embeds and fp32_text_embeds", torch.dist(fp32_image_embeds,fp32_text_embeds,p=1)/len(fp32_text_embeds))

print("L2 distance between fp32_text_embeds and int8_text_embeds", torch.sqrt(torch.dist(fp32_text_embeds,int8_text_embeds,p=2)**2/len(fp32_text_embeds)))
print("L2 distance between fp32_image_embeds and int8_text_embeds", torch.sqrt(torch.dist(fp32_image_embeds,int8_text_embeds,p=2)**2/len(fp32_text_embeds)))
print("L2 distance between fp32_image_embeds and fp32_text_embeds", torch.sqrt(torch.dist(fp32_image_embeds,fp32_text_embeds,p=2)**2/len(fp32_text_embeds)))


# large14   1000
# fp32_loss 16.3116
# int8_loss 14.2129
# L1 distance between fp32_text_embeds and int8_text_embeds  2.9399
# L1 distance between fp32_image_embeds and int8_text_embeds  25.2129
# L1 distance between fp32_image_embeds and fp32_text_embeds  25.2922
# L2 distance between fp32_text_embeds and int8_text_embeds  0.1423
# L2 distance between fp32_image_embeds and int8_text_embeds  1.3508
# L2 distance between fp32_image_embeds and fp32_text_embeds  1.3522

# base16    1000
# fp32_loss 15.1359
# int8_loss 12.6957
# L1 distance between fp32_text_embeds and int8_text_embeds  3.4156
# L1 distance between fp32_image_embeds and int8_text_embeds  18.7959
# L1 distance between fp32_image_embeds and fp32_text_embeds  19.2942
# L2 distance between fp32_text_embeds and int8_text_embeds  0.2078
# L2 distance between fp32_image_embeds and int8_text_embeds  1.2956
# L2 distance between fp32_image_embeds and fp32_text_embeds  1.3029