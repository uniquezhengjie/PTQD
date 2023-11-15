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

def prepare_calibration_data2(dataloader, init_steps):
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
                pixel_values.append(batch["pixel_values"].to("cuda"))
                input_ids.append(batch["input_ids"].to("cuda"))
                attention_mask.append(batch["attention_mask"].to("cuda"))
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

def prepare_dataset_val(opt_init_steps=1000):
    """
    Prepares a vision-text dataset for quantization.
    """
    dataset = load_dataset("conceptual_captions", streaming=True)
    dataloader = torch.utils.data.DataLoader(dataset["validation"], collate_fn=collate_fn, batch_size=1)
    calibration_data = prepare_calibration_data2(dataloader, opt_init_steps)
    return calibration_data

def prepare_dataset2(inputs):
    data = []
    data.append(
    {
        "pixel_values": inputs["pixel_values"].to("cuda"),
        "input_ids": inputs["input_ids"].to("cuda"),
        "attention_mask": inputs["attention_mask"].to("cuda"),
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

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
# fp16_model_path = 'clip-vit-base-patch16.xml'

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
max_length = model.config.text_config.max_position_embeddings
fp16_model_path = 'clip-vit-large-patch14.xml'

model.cuda()
model.eval()


if False:
    calibration_data = prepare_dataset_val()
    text_embeds = []
    image_embeds = []
    for input_data in calibration_data:
        torch.cuda.empty_cache()
        print(input_data["pixel_values"].shape)
        with torch.no_grad():
            results = model(**input_data)
            text_embeds.append(results['text_embeds'].cpu())
            image_embeds.append(results['image_embeds'].cpu())

    # print(text_embeds)
    with torch.no_grad():
        text_embeds = torch.concat(text_embeds)
        print(text_embeds.shape)
        image_embeds = torch.concat(image_embeds)
        logit_scale = model.logit_scale.exp().cpu()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        print(logit_scale)
        loss = clip_loss(logits_per_text)
        print(loss)
        # 15.1359

if False:
    results = model(**calibration_data[0])
    logits_per_image = results['logits_per_image']  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()  # we can take the softmax to get the label probabilities
    # print(probs)
    logits_per_text = results['logits_per_text'] # this is the image-text similarity score
    probs = logits_per_text.softmax(dim=0).detach().cpu().numpy()  # we can take the softmax to get the label probabilities
    print(probs)
    logits_per_text = results['logits_per_text']
    loss = clip_loss(logits_per_text)
    print(loss)

if True:

    image = Image.open('/home/reexen/projects/tiny-stable-diffusion/outputs/txt2img-samples/a_dog_under_the_tree/seed_939062_00002.png')

    input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]
    inputs = processor(text=text_descriptions, images=[image]*10, return_tensors="pt", padding="max_length")
    # inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)
    # calibration_data = prepare_dataset()
    print(inputs)
    model.cpu()
    results = model(**inputs)
    model.config.torchscript = True

    ov_model = ov.convert_model(model, example_input=dict(inputs))
    ov.save_model(ov_model, fp16_model_path)

