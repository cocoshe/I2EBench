import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def run_clip(text, img_path):
    print(text)
    print(img_path)
    
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, _ = model(image, text)

    return float(logits_per_image[0][0])
