import torch
from PIL import Image
from torchvision import transforms
from model import ImageStoryGenerator

def generate_story(image_path, model_path="vlm_model.pt", max_length=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageStoryGenerator()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    input_ids = None
    attention_mask = None
    generated = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(image, input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            if input_ids is None:
                input_ids = next_token
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)

            generated.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break

    story = tokenizer.decode(generated, skip_special_tokens=True)
    print("Generated Story:\n", story)

if __name__ == "__main__":
    import sys
    generate_story(sys.argv[1])
