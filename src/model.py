import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torchvision.models as models

class CLIPImageEncoder(nn.Module):
    def __init__(self, embed_size=512):
        super(CLIPImageEncoder, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
    
    def forward(self, images):
        return self.cnn(images)

class ImageStoryGenerator(nn.Module):
    def __init__(self, embed_size=512, gpt2_model='gpt2'):
        super(ImageStoryGenerator, self).__init__()
        self.encoder = CLIPImageEncoder(embed_size)
        self.mapper = nn.Linear(embed_size, 768)  # GPT-2 hidden size
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    
    def forward(self, images, input_ids=None, attention_mask=None):
        with torch.no_grad():
            img_features = self.encoder(images)
        mapped_features = self.mapper(img_features).unsqueeze(1)
        
        if input_ids is not None:
            inputs_embeds = self.gpt2.transformer.wte(input_ids)
            inputs_embeds = torch.cat([mapped_features, inputs_embeds], dim=1)
            attention_mask = torch.cat([torch.ones((input_ids.size(0), 1), device=input_ids.device), attention_mask], dim=1)
        else:
            inputs_embeds = mapped_features
            attention_mask = torch.ones((images.size(0), 1), device=images.device)
        
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs
