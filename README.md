# Image-to-Story Generator

A Vision-Language Model (VLM) that takes an image as input and generates a short, coherent, and imaginative story based on its visual content.

## Overview

This project combines a vision encoder (ResNet50) with a language model (GPT-2) to form a lightweight, powerful VLM pipeline. A linear projection bridges the vision features to GPT-2's embedding space, enabling creative and descriptive story generation from images.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/CLIP-GPT_diagram.png/640px-CLIP-GPT_diagram.png" width="600" />
</p>

## Features

-  Vision encoder using pre-trained ResNet50
-  GPT-2 decoder for text generation
-  Learnable mapper between image and text embeddings
-  Simple training loop on custom datasets
-  Generates creative stories from any image
-  Plug-and-play inference with a single command

---

##  Project Structure

```
image-to-story-generator/
├── requirements.txt # includes all of the required dependencies            
├── README.MD # includes instructions for setup and usage
└── src/
    └── model.py # Core model combining encoder + GPT2
    └── dataset.py # Custom dataset class
    └── train.py # Training script
    └── generate.py # Inference script
    └── utils.py # utility functions
└── data/
    ├── images/ # Folder with input images
    └── stories.txt # Tab-separated image-to-story data
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/image-to-story.git
cd image-to-story
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Put your images in the `data/images/` folder.

Create a `data/stories.txt` file (tab-separated format):

```
dog.jpg    A small dog runs through a sunny field, chasing butterflies.
cat.jpg    A curious cat climbs a tree, watching birds fly above.
car.jpg    A red sports car speeds down a desert highway under the setting sun.
```

Make sure image names match exactly.

---

## Training

```bash
python train.py
```

Model will be saved as `vlm_model.pt`.

---

## Generating Stories

```bash
python generate.py data/images/dog.jpg
```

Output:
```
Generated Story:
A playful puppy dashes across the green meadow, thrilled by the fluttering butterflies.
```

---

## Model Details

| Component         | Model           |
|------------------|-----------------|
| Image Encoder     | ResNet-50 (pretrained) |
| Language Decoder | GPT-2 (small)   |
| Mapping Layer    | Linear          |
| Tokenizer        | GPT-2 Tokenizer |
| Image Size       | 224x224         |



---

