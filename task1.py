import os
import re
import json
import argparse

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

import torch
import torchvision.transforms as T

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoProcessor, CLIPModel, Kosmos2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
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

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the area of union
    unionArea = boxAArea + boxBArea - interArea

    # Compute the IoU
    iou = interArea / float(unionArea)

    return iou

def compute_sim(image1, image2, processor_clip, model_clip, device):
    image_tensor1 = processor_clip(images=image1, return_tensors="pt").to(device)
    image_tensor2 = processor_clip(images=image2, return_tensors="pt").to(device)

    image_feature1 = model_clip.get_image_features(image_tensor1.pixel_values)
    image_feature2 = model_clip.get_image_features(image_tensor2.pixel_values)

    similarity = torch.nn.functional.cosine_similarity(image_feature1, image_feature2, dim=-1)

    return similarity.item()

def generate_bbox(input, model_ko, processor_ko):
    generated_ids = model_ko.generate(
        pixel_values=input["pixel_values"],
        input_ids=input["input_ids"],
        attention_mask=input["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=input["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=64,
    )

    generated_text = processor_ko.batch_decode(generated_ids, skip_special_tokens=True)[0]
    _, entities = processor_ko.post_process_generation(generated_text)
    return entities

def task1(item, model_clip, processor_clip, tokenizer, nli_model, model_ko, processor_ko, model_sentence, base, device, args):
    
    img_path = item['img_local_path']
    local_path = args.path

    cap1_bboxes = []
    cap2_bboxes = []
    iou_list=[]

    nil_result = nli(item['caption1'], item['caption2'], tokenizer, nli_model, device)
    if nil_result['entailment'] >= 75:
        return 0
    elif nil_result['contradiction'] >= 75:
        return 1
    
    # Compute Semantic Textual Similarity
    embeddings1 = model_sentence.encode(item['caption1'], convert_to_tensor=True)
    embeddings2 = model_sentence.encode(item['caption2'], convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    if np.mean([cosine_scores.item(), float(item['bert_base_score'])]) <= 0.5:
        return 1
    
    # Load Image for Stable diffusion and Image captioning
    original_image = Image.open(os.path.join(local_path, img_path))

    # Stable diffusion 
    generator = torch.Generator(device=device).manual_seed(1024)
    generated_image1 = base(
    prompt=item['caption1'],
    image=original_image, strength=0.75, guidance_scale=10, generator=generator
    ).images[0]

    generated_image2 = base(
    prompt=item['caption2'],
    image=original_image, strength=0.75, guidance_scale=10, generator=generator
    ).images[0]

    similarity_stable = compute_sim(generated_image1, generated_image2, processor_clip, model_clip, device)

    if similarity_stable>=0.8:
        return 0

    # Image captioning (Extract bboxes based on Caption1, 2)
    prompt_ko1 = f"<grounding><phrase>{item['caption1']}</phrase>"
    prompt_ko2 = f"<grounding><phrase>{item['caption2']}</phrase>"

    inputs1 = processor_ko(text=prompt_ko1, images=original_image, return_tensors="pt").to(device)
    inputs2 = processor_ko(text=prompt_ko2, images=original_image, return_tensors="pt").to(device)

    entities1 = generate_bbox(inputs1, model_ko, processor_ko)
    entities2 = generate_bbox(inputs2, model_ko, processor_ko)

    for _, _, bboxes in entities1:
        cap1_bboxes.append(bboxes[0])
    for _, _, bboxes in entities2:
        cap2_bboxes.append(bboxes[0])

    # Compute # of matched bboxes 
    for (x1_norm, y1_norm, x2_norm, y2_norm) in cap1_bboxes:
        for (x1_norm2, y1_norm2, x2_norm2, y2_norm2) in cap2_bboxes:
            iou_list.append(calculate_iou([x1_norm, y1_norm, x2_norm, y2_norm],[x1_norm2, y1_norm2, x2_norm2, y2_norm2]))
    count_matched = sum(1 for iou in iou_list if iou > 0.75)

    if count_matched == 0:
        return 1
    else:
        return 0

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

    # NLI Model
    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nli_model.to(device)

    # KOSMOS Captioning Model
    model_ko = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)
    processor_ko = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    # Semantic Similarity Model
    model_sentence = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    # Stable diffution Model 
    base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    for i, item in enumerate(modified_data):
        context_label = item['context_label']
        prediction = task1(item, model_clip=model_clip, processor_clip=processor_clip, tokenizer=tokenizer, nli_model=nli_model, model_ko=model_ko,
                           processor_ko=processor_ko, model_sentence=model_sentence, base=base, device=device, args=args)
        prediction_list.append(prediction)
        true_labels.append(context_label)
    accuracy = accuracy_score(true_labels, prediction_list)
    average_precision = average_precision_score(true_labels, prediction_list)
    f1 = f1_score(true_labels, prediction_list)
    
    print("Accuracy:", accuracy)
    print("Average Precision:", average_precision)
    print("F1-Score:", f1)   

    # Save the Result
    output_file = "task1_result.txt"
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
    