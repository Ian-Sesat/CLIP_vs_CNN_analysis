"""
Universal Embedding Extractor
------------------------------
Extracts train, val and test embeddings for any model.
Imports model and dataloaders from the corresponding loader file.

Usage:
    Set MODEL_TYPE to desired model and run.
    Embeddings saved to SAVE_DIR as .npz file.

Supported models:
    resnet50        : ResNet50 + Attention Head
    efficientnet_b1 : EfficientNet-B1 + Attention Head
    clip            : OpenCLIP ViT-H/14 zero-shot
    siglip2         : SigLIP2 So400m zero-shot
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

# CONFIG 
MODEL_TYPE = 'resnet50'   # resnet50 | efficientnet_b1 | clip | siglip2

SAVE_DIR   = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'

EMBEDDINGS = {
    'resnet50'        : os.path.join(SAVE_DIR, 'embeddings_resnet50_best.npz'),
    'efficientnet_b1' : os.path.join(SAVE_DIR, 'embeddings_effnetb1_attention.npz'),
    'clip'            : os.path.join(SAVE_DIR, 'embeddings_clip_vith14_zeroshot.npz'),
    'siglip2'         : os.path.join(SAVE_DIR, 'embeddings_siglip2_zeroshot.npz'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# LOAD MODEL AND DATALOADERS FROM LOADER 
if MODEL_TYPE == 'resnet50':
    from resnet50_loader import model, train_loader, val_loader, test_loader

elif MODEL_TYPE == 'efficientnet_b1':
    from efficientnet_loader import model, train_loader, val_loader, test_loader

elif MODEL_TYPE == 'clip':
    from clip_loader import model, train_loader, val_loader, test_loader

elif MODEL_TYPE == 'siglip2':
    from siglip2_loader import model, train_loader, val_loader, test_loader

else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

print(f"Model and dataloaders loaded for: {MODEL_TYPE}")


# EXTRACTION FUNCTIONS 
def extract_standard(loader, model, device):
    # For ResNet50 and EfficientNet — uses return_embedding flag
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)
            with autocast():
                embeddings = model(images, return_embedding=True)
            all_embeddings.append(embeddings.cpu().float().numpy())
            all_labels.append(labels.numpy())

    return (np.concatenate(all_embeddings, axis=0),
            np.concatenate(all_labels,     axis=0))


def extract_clip(loader, model, device):
    # For OpenCLIP — uses encode_image and L2 normalisation
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)
            with autocast():
                embeddings = model.encode_image(images)
                embeddings = nn.functional.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().float().numpy())
            all_labels.append(labels.numpy())

    return (np.concatenate(all_embeddings, axis=0),
            np.concatenate(all_labels,     axis=0))


def extract_siglip2(loader, model, device):
    # For SigLIP2 — uses vision_model pooler_output and L2 normalisation
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            if batch is None:
                continue
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            with autocast():
                outputs    = model.vision_model(pixel_values=pixel_values)
                embeddings = outputs.pooler_output
                embeddings = nn.functional.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().float().numpy())
            all_labels.append(labels.numpy())

    return (np.concatenate(all_embeddings, axis=0),
            np.concatenate(all_labels,     axis=0))


# Select extraction function based on model type
extract_fn = {
    'resnet50'        : extract_standard,
    'efficientnet_b1' : extract_standard,
    'clip'            : extract_clip,
    'siglip2'         : extract_siglip2,
}[MODEL_TYPE]


# EXTRACT EMBEDDINGS 
save_path = EMBEDDINGS[MODEL_TYPE]

print(f"\nExtracting embeddings for: {MODEL_TYPE}")
torch.cuda.empty_cache()

print("Extracting train embeddings ...")
train_embeddings, train_labels = extract_fn(train_loader, model, device)

print("Extracting val embeddings ...")
val_embeddings,   val_labels   = extract_fn(val_loader,   model, device)

print("Extracting test embeddings ...")
test_embeddings,  test_labels  = extract_fn(test_loader,  model, device)

print(f"Train : {train_embeddings.shape}")
print(f"Val   : {val_embeddings.shape}")
print(f"Test  : {test_embeddings.shape}")

np.savez(save_path,
         train_embeddings=train_embeddings, train_labels=train_labels,
         val_embeddings=val_embeddings,     val_labels=val_labels,
         test_embeddings=test_embeddings,   test_labels=test_labels)

print(f"Embeddings saved to {save_path}")