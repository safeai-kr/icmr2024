import os
import re
import json
import argparse

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

import torch
import torchvision.transforms as T

from transformers import AutoProcessor, CLIPModel
from diffusers import StableDiffusionXLImg2ImgPipeline

def convert_to_lowercase(text):
    return text.lower()

def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def nli(str1, str2, tokenizer, nli_model, device):
    inputs = tokenizer(str1, str2, truncation=True, return_tensors="pt").to(device)
    inputs.to(device)

    outputs = nli_model(**inputs)
    prediction = torch.softmax(outputs.logits[0], -1).tolist()

    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}

    return prediction

def compute_sim(image1, image2, processor_clip, model_clip, device):
    image_tensor1 = processor_clip(images=image1, return_tensors="pt").to(device)
    image_tensor2 = processor_clip(images=image2, return_tensors="pt").to(device)

    image_feature1 = model_clip.get_image_features(image_tensor1.pixel_values)
    image_feature2 = model_clip.get_image_features(image_tensor2.pixel_values)

    similarity = torch.nn.functional.cosine_similarity(image_feature1, image_feature2, dim=-1)

    return similarity.item()

def task2(item, model_clip, processor_clip, base, device, args):
    
    img_path = item['img_local_path']
    local_path = args.path
    
    # Load Image for Stable diffusion and Image captioning
    original_image = Image.open(os.path.join(local_path, img_path))

    # Stable diffusion 
    generator = torch.Generator(device=device).manual_seed(1024)
    generated_image = base(
    prompt=item['caption1'],
    image=original_image, strength=0.75, guidance_scale=10, generator=generator
    ).images[0]

    similarity_stable = compute_sim(original_image, generated_image, processor_clip, model_clip, device)
    
    if similarity_stable>=0.8:
        return 0
    else:
        return 1

def main(args):
    test_file = f"{args.path}/test.json" 

    test_data = []
    modified_data = []
    prediction_list = []
    true_labels = []

    with open(test_file, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                test_data.append(json_obj)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line.strip()}")

    for json_obj in test_data:
        # Remove unused column
        if 'article_url' in json_obj:
            del json_obj['article_url']
        if 'entity_list' in json_obj:
            del json_obj['entity_list']
        if 'maskrcnn_bboxes' in json_obj:
            del json_obj['maskrcnn_bboxes']
        if 'bert_large_score' in json_obj:
            del json_obj['bert_large_score']
        
        # Pre-processing
        if 'caption1' in json_obj:
            cap1_m = remove_special_characters(json_obj['caption1'])
            cap1_m = convert_to_lowercase(cap1_m)
        if 'caption2' in json_obj:
            cap2_m = remove_special_characters(json_obj['caption2'])
            cap2_m = convert_to_lowercase(cap2_m)
        
        modified_data.append(json_obj)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP Model
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Stable diffution Model 
    base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    for i, item in enumerate(modified_data):
        context_label = item['context_label']
        prediction = task2(item, model_clip=model_clip, processor_clip=processor_clip, base=base, device=device, args=args)
        prediction_list.append(prediction)
        true_labels.append(context_label)
        
    accuracy = accuracy_score(true_labels, prediction_list)
    average_precision = average_precision_score(true_labels, prediction_list)
    f1 = f1_score(true_labels, prediction_list)
    
    print("Accuracy:", accuracy)
    print("Average Precision:", average_precision)
    print("F1-Score:", f1)   

    # Save the Result
    output_file = "task2_result.txt"
    with open(output_file, 'w') as file:
        for item in prediction_list:
            file.write(str(item) + '\n')
    print(f'The result has been saved to {output_file}')

    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path to the dataset")
    args = parser.parse_args()
    
    main(args)
    