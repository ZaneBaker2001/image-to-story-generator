import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ImageStoryGenerator
from dataset import ImageStoryDataset
from transformers import GPT2Tokenizer

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    dataset = ImageStoryDataset("data/images", "data/stories.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ImageStoryGenerator()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(5):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for images, input_ids, attention_mask in loop:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "vlm_model.pt")

if __name__ == "__main__":
    train()
