import os
import numpy as np
import warnings
import faiss

from tqdm import tqdm

warnings.filterwarnings("ignore")

SAVE_DIR        = '/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'
EMBEDDINGS_PATH = os.path.join(SAVE_DIR, 'embeddings_resnet50_best.npz')
MODEL_NAME      = 'ResNet50 (full + attention)'

# Available embeddings:
# embeddings_resnet50_best.npz
# embeddings_effnetb1_attention.npz
# embeddings_clip_vith14_zeroshot.npz
# embeddings_siglip2_zeroshot.npz

NUM_CLASSES = 100

print(f"Loading : {EMBEDDINGS_PATH}")
data             = np.load(EMBEDDINGS_PATH)
train_embeddings = data['train_embeddings']
train_labels     = data['train_labels']
val_embeddings   = data['val_embeddings']
val_labels       = data['val_labels']
test_embeddings  = data['test_embeddings']
test_labels      = data['test_labels']

print(f"Train : {train_embeddings.shape}")
print(f"Val   : {val_embeddings.shape}")
print(f"Test  : {test_embeddings.shape}")


def build_faiss_index(embeddings):
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def evaluate_map(test_embeddings, test_labels, index, train_labels):
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    aps        = []
    chunk_size = 1000

    for start in tqdm(range(0, len(query), chunk_size), desc="mAP"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        _, indices = index.search(chunk, index.ntotal)

        for i, idx in enumerate(indices):
            query_label    = test_labels[start + i]
            retrieved      = train_labels[idx]
            total_relevant = (train_labels == query_label).sum()

            ap      = 0.0
            correct = 0

            for rank, label in enumerate(retrieved, 1):
                if label == query_label:
                    correct += 1
                    ap      += correct / rank

            ap = ap / total_relevant if total_relevant > 0 else 0.0
            aps.append(ap)

    map_score = np.mean(aps)
    print(f"  mAP : {map_score:.4f}")
    return map_score


def evaluate_recall(test_embeddings, test_labels, index, train_labels, k=1):
    query      = test_embeddings.astype('float32')
    faiss.normalize_L2(query)
    correct    = 0
    chunk_size = 10000

    for start in tqdm(range(0, len(query), chunk_size), desc=f"Recall@{k}"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        _, indices = index.search(chunk, k)

        for i, idx in enumerate(indices):
            query_label  = test_labels[start + i]
            top_k_labels = train_labels[idx]
            if query_label in top_k_labels:
                correct += 1

    recall = correct / len(test_embeddings)
    print(f"  Recall@{k} : {recall:.4f}")
    return recall


print(f"\nEVALUATION — {MODEL_NAME}")
print("=" * 50)

index     = build_faiss_index(train_embeddings)
map_score = evaluate_map(test_embeddings, test_labels, index, train_labels)
recall_1  = evaluate_recall(test_embeddings, test_labels, index, train_labels, k=1)
average   = (map_score + recall_1) / 2

print(f"\nFINAL RESULTS — {MODEL_NAME}")
print("=" * 50)
print(f"\n{'Metric':<25} {'Score':>10}")
print("-" * 35)
print(f"{'mAP':<25} {map_score*100:>9.2f}%")
print(f"{'Recall@1':<25} {recall_1*100:>9.2f}%")
print(f"{'Average':<25} {average*100:>9.2f}%")
print("-" * 35)