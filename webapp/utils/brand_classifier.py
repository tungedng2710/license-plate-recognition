import numpy as np
import torch
from PIL import Image
import open_clip

# Load model and tokenizer
model_id = "ViT-B-16"
pretrained_id = "laion2b_s34b_b88k"
model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained_id)
model.eval()
tokenizer = open_clip.get_tokenizer(model_id)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
brands = ['Audi', 'Chrysler', 'Citroen', 'GMC', 'Honda', 'Hyundai', 
          'Mazda', 'Mercedes', 'Mercury', 'Mitsubishi', 
          'Nissan', 'Renault', 'Toyota', 'Volkswagen', 
          'bmw', 'cadillac', 'chevrolet', 'ford', 'kia', 'lexus', 
          'lincoln', 'mini', 'no class', 'porsche', 'range rover', 
          'subaru', 'suzuki', 'volvo']

def argmax(listing):
    return np.argmax(listing)

def process_image(image_pil):
    image = preprocess(image_pil).unsqueeze(0).to(device)
    return image

def process_text(prompts):
    text = tokenizer(prompts).to(device)
    return text

def recognize_brand(image):    
    # Process image and text
    image = process_image(image)
    text = process_text(brands)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # print("Label probs:", brands[argmax(text_probs.cpu()[0])])  # prints probabilities
    return brands[argmax(text_probs.cpu()[0])]