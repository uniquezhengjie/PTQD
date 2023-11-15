import openvino as ov
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
from PIL import Image

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
# fp16_model_path = 'clip-vit-base-patch16.xml'

model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
max_length = model.config.max_position_embeddings
processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
fp16_model_path = 'clip-vit-large-patch14_txt.xml'

input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

inputs = processor(text_descriptions, truncation=True, max_length=max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

results = model(inputs["input_ids"])
print(results['last_hidden_state'].shape)
# print(results['pooler_output'].shape)

# model.config.torchscript = True
# ov_model = ov.convert_model(model, example_input=inputs["input_ids"])
# ov.save_model(ov_model, fp16_model_path)