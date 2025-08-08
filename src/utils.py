import torch
from torchvision import transforms
from PIL import Image
import os


def load_image(image_path, image_size=(224, 224)):
    """
    Load and preprocess an image.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # shape: (1, C, H, W)


def save_model(model, path):
    """
    Save model state_dict.
    """
    torch.save(model.state_dict(), path)
    print(f"[✓] Model saved to {path}")


def load_model(model, path, device):
    """
    Load model state_dict.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[✓] Model loaded from {path}")
    return model


def decode_tokens(tokenizer, token_ids):
    """
    Decode a list of token IDs to text, skipping special tokens.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def ensure_dir(path):
    """
    Create directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
