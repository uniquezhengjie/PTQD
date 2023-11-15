import openvino as ov
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import torch
import torch.nn as nn
from datasets import load_dataset
import logging
import sys,os
import numpy as np
from openvino.runtime import Core, serialize
from openvino.runtime import compile_model
from tqdm import tqdm
import nncf

save_count = 0
CLASS_NUM = 1000
IMAGE_PER_CLASS = 1


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


def prepare_calibration_data_val(dataloader, init_steps, device="cuda"):
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
    print('len dataset:', len(dataloader))
    for img, text in dataloader.items():
        if counter % 100 == 0:
            print(counter)
        if counter == init_steps:
            break
        counter += 1
        image = Image.open(img)
        image = image.convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding="max_length")
        with torch.no_grad():
            pixel_values.append(inputs["pixel_values"].to(device))
            input_ids.append(inputs["input_ids"].to(device))
            attention_mask.append(inputs["attention_mask"].to(device))

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


def prepare_dataset_val(opt_init_steps=2000, device="cuda"):
    """
    Prepares a vision-text dataset for quantization.
    """
    label_txt = '/data1/datasets/Imagenet-ori/caffe_labels/synset_words.txt'
    image_dir = '/data1/datasets/Imagenet-ori/train'
    with open(label_txt, 'r') as f:
        code2text = f.read()

    # code2text = [i for i in code2text.strip().split("\n") if len(i.split(','))==1]
    code2text = [i.split(',')[0] for i in code2text.strip().split("\n")]
    code2text = {i.split()[0]:' '.join(i.split()[1:]) for i in code2text}

    dataset = {}

    for c, t in code2text.items():
        sub_dir = os.path.join(image_dir,c)
        for img in os.listdir(sub_dir)[:IMAGE_PER_CLASS]:
            img_path = os.path.join(sub_dir, img)
            dataset[img_path] = "This is a photo of a {}".format(t)
    
    calibration_data = prepare_calibration_data_val(dataset, opt_init_steps, device=device)
    return calibration_data

def prepare_calibration_data(dataloader, init_steps, device="cuda"):
    """
    This function prepares calibration data from a dataloader for a specified number of initialization steps.
    It iterates over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    print(f"Fetching {init_steps} for the initialization...")
    counter = 0
    print('len dataset:', len(dataloader))
    for img, text in dataloader.items():
        if counter % 100 == 0:
            print(counter)
        if counter == init_steps:
            break
        counter += 1
        image = Image.open(img)
        image = image.convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding="max_length")
        with torch.no_grad():
            data.append(
                {
                        "pixel_values": inputs["pixel_values"].to("cpu"),
                        "input_ids": inputs["input_ids"].to("cpu"),
                        "attention_mask": inputs["attention_mask"].to("cpu")
                }
            )
    return data

def prepare_dataset(opt_init_steps=2000, device="cuda"):
    """
    Prepares a vision-text dataset for quantization.
    """
    label_txt = '/data1/datasets/Imagenet-ori/caffe_labels/synset_words.txt'
    image_dir = '/data1/datasets/Imagenet-ori/train'
    with open(label_txt, 'r') as f:
        code2text = f.read()

    # code2text = [i for i in code2text.strip().split("\n") if len(i.split(','))==1]
    code2text = [i.split(',')[0] for i in code2text.strip().split("\n")]
    code2text = {i.split()[0]:' '.join(i.split()[1:]) for i in code2text}

    dataset = {}

    for c, t in code2text.items():
        sub_dir = os.path.join(image_dir,c)
        for img in os.listdir(sub_dir)[-IMAGE_PER_CLASS:]:
            img_path = os.path.join(sub_dir, img)
            dataset[img_path] = "This is a photo of a {}".format(t)
    
    calibration_data = prepare_calibration_data(dataset, opt_init_steps, device=device)
    return calibration_data


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def gt_class(c, p):
    res = []
    for i in range(c):
        for _ in range(p):
            res.append(i)
    return np.array(res)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
max_length = model.config.text_config.max_position_embeddings
fp16_model_path = 'clip-vit-base-patch16.xml'
int8_model_path = 'clip-vit-base-patch16_imagenet_int8_torch.xml'

# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# max_length = model.config.text_config.max_position_embeddings
# fp16_model_path = 'clip-vit-large-patch14.xml'
# int8_model_path = 'clip-vit-large-patch14_imagenet_int8.xml'

model.cuda()
model.eval()

# image = Image.open('/home/reexen/projects/tiny-stable-diffusion/outputs/txt2img-samples/a_dog_under_the_tree/seed_939062_00002.png')

# input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
# text_descriptions = [f"This is a photo of a {label}" for label in input_labels]
# inputs = processor(text=text_descriptions, images=[image]*10, return_tensors="pt", padding="max_length")
# inputs['return_loss'] = True 

if True:
    core = Core()
    nncf.set_log_level(logging.ERROR)
    ov_model = core.read_model(fp16_model_path)
    calibration_data = prepare_dataset(opt_init_steps=CLASS_NUM*IMAGE_PER_CLASS)
    calibration_dataset = nncf.Dataset(calibration_data)
    quantized_model = nncf.quantize(
        model=ov_model,
        # model=model,
        calibration_dataset=calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
    )

    serialize(quantized_model, int8_model_path)

if True:
    calibration_data = prepare_dataset_val(opt_init_steps=CLASS_NUM*IMAGE_PER_CLASS)
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
        # print(fp32_loss)
        # 3.4700
        logits_per_image = logits_per_text.t()[:,::IMAGE_PER_CLASS] 
        probs = logits_per_image.softmax(dim=1).detach().numpy()
        pred_class = np.argmax(probs,1)
        fp32_acc = sum(pred_class==gt_class(CLASS_NUM,IMAGE_PER_CLASS))/len(pred_class)
        

if True:
    calibration_data = prepare_dataset_val(opt_init_steps=CLASS_NUM*IMAGE_PER_CLASS, device="cpu")
    compiled_model = compile_model(int8_model_path)
    # logits_per_text = compiled_model.output(1)
    text_embed_out = compiled_model.output(2)
    image_embed_out = compiled_model.output(3)

    text_embeds = []
    image_embeds = []
    for input_data in tqdm(calibration_data):
        # ov_logits_per_text = compiled_model(calibration_data[0])[logits_per_text]
        # probs = softmax(ov_logits_per_text, axis=0)
        text_embed = compiled_model(input_data)[text_embed_out]
        image_embed = compiled_model(input_data)[image_embed_out]
        text_embeds.append(torch.from_numpy(text_embed))
        image_embeds.append(torch.from_numpy(image_embed))
        # print(probs)
        # loss = clip_loss(torch.from_numpy(ov_logits_per_text))
        # print(loss)

    with torch.no_grad():
        int8_text_embeds = torch.concat(text_embeds)
        int8_image_embeds = torch.concat(image_embeds)
        logit_scale = model.logit_scale.exp().cpu()
        logits_per_text = torch.matmul(int8_text_embeds, int8_image_embeds.t()) * logit_scale
        int8_loss = clip_loss(logits_per_text)
        logits_per_image = logits_per_text.t()[:,::IMAGE_PER_CLASS] 
        probs = logits_per_image.softmax(dim=1).detach().numpy()
        pred_class = np.argmax(probs,1)
        int8_acc = sum(pred_class==gt_class(CLASS_NUM,IMAGE_PER_CLASS))/len(pred_class)


print("fp32_loss: ",fp32_loss)
print("int8_loss: ",int8_loss)

print("fp32_acc: ",fp32_acc)
print("int8_acc: ",int8_acc)

print("L1 distance between fp32_text_embeds and int8_text_embeds", torch.dist(fp32_text_embeds,int8_text_embeds,p=1)/len(fp32_text_embeds))
print("L1 distance between fp32_image_embeds and int8_text_embeds", torch.dist(fp32_image_embeds,int8_text_embeds,p=1)/len(fp32_text_embeds))
print("L1 distance between fp32_image_embeds and fp32_text_embeds", torch.dist(fp32_image_embeds,fp32_text_embeds,p=1)/len(fp32_text_embeds))

print("L2 distance between fp32_text_embeds and int8_text_embeds", torch.sqrt(torch.dist(fp32_text_embeds,int8_text_embeds,p=2)**2/len(fp32_text_embeds)))
print("L2 distance between fp32_image_embeds and int8_text_embeds", torch.sqrt(torch.dist(fp32_image_embeds,int8_text_embeds,p=2)**2/len(fp32_text_embeds)))
print("L2 distance between fp32_image_embeds and fp32_text_embeds", torch.sqrt(torch.dist(fp32_image_embeds,fp32_text_embeds,p=2)**2/len(fp32_text_embeds)))

# CLASS_NUM = 1000
# IMAGE_PER_CLASS = 10
# large14  
# fp32_loss 4.2327
# int8_loss 4.3577
# fp32_acc 0.7146
# int8_acc 0.6902
# L1 distance between fp32_text_embeds and int8_text_embeds  4.4829
# L1 distance between fp32_image_embeds and int8_text_embeds  21.4428
# L1 distance between fp32_image_embeds and fp32_text_embeds  21.6436
# L2 distance between fp32_text_embeds and int8_text_embeds  0.2166
# L2 distance between fp32_image_embeds and int8_text_embeds  1.2201
# L2 distance between fp32_image_embeds and fp32_text_embeds  1.2247

# base16
# fp32_loss 4.3480
# int8_loss 4.5888
# fp32_acc 0.654
# int8_acc 0.5966
# L1 distance between fp32_text_embeds and int8_text_embeds  4.0469
# L1 distance between fp32_image_embeds and int8_text_embeds  16.1127
# L1 distance between fp32_image_embeds and fp32_text_embeds  16.6194
# L2 distance between fp32_text_embeds and int8_text_embeds  0.2463
# L2 distance between fp32_image_embeds and int8_text_embeds  1.1815
# L2 distance between fp32_image_embeds and fp32_text_embeds  1.1838

# CLASS_NUM = 1000
# IMAGE_PER_CLASS = 1

# base16
# fp32_loss 1.6305
# int8_loss 1.9626

# large14  
# fp32_loss 1.4548
# int8_loss 1.5530

