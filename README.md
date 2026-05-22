# Image Retrieval

Extracts embeddings from four models and evaluates image retrieval performance using mAP.

## Pipeline

**Step 1 — Extract embeddings** (`extractor.py`)

Set `MODEL_TYPE` to the desired model and run. Imports the model and dataloaders from the corresponding loader file and saves train, val and test embeddings to disk as `.npz`.

```python
MODEL_TYPE = 'resnet50'  # resnet50 | efficientnet_b1 | clip | siglip2
```

**Step 2 — Evaluate** (`image_retrieval_evaluation.py`)

Set `EMBEDDINGS_PATH` to the desired model embeddings and run. Builds a FAISS index on train embeddings and evaluates mAP over full gallery retrieval against the test set.

```python
EMBEDDINGS_PATH = os.path.join(SAVE_DIR, 'embeddings_resnet50_best.npz')
MODEL_NAME      = 'ResNet50 (full + attention)'
```

## Results

Fine-tuned CNNs significantly outperform zero-shot Transformers on mAP. ResNet50 achieves 84.33% compared to 61.38% for SigLIP2 — a gap of 22.95 percentage points — despite having 16× fewer parameters. EfficientNet-B1 achieves competitive performance at 83.06% with only 7.8M parameters. Zero-shot Transformers, while strong on semantic tasks, produce embeddings that are less discriminative for visual similarity ranking without fine-tuning.

| Model | mAP |
|-------|-----|
| ResNet50 (full + attention) | 84.33% |
| EfficientNet-B1 (attention) | 83.06% |
| OpenCLIP ViT-H/14 | 63.84% |
| SigLIP2 So400m | 61.38% |
