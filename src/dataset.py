from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from transformers import GPT2Tokenizer

class ImageStoryDataset(Dataset):
    def __init__(self, image_folder, story_file, tokenizer, max_length=100):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(story_file, 'r') as f:
            lines = f.readlines()
        self.samples = [line.strip().split('\t') for line in lines]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, story = self.samples[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = self.transform(Image.open(img_path).convert('RGB'))
        tokens = self.tokenizer(story, return_tensors='pt', padding="max_length", truncation=True, max_length=self.max_length)
        return image, tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)
