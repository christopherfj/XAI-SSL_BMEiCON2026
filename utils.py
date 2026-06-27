import os
import pickle
import numpy as np
from PIL import Image
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import KFold, train_test_split
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import shap

SEED = 42

DATASET_CLASSES = {
    "kidney": ["Cyst", "Normal", "Stone", "Tumor"],
    "covid":  ["COVID", "Normal", "Viral Pneumonia"],
}

XAI_LEVELS      = [0, 10, 25, 50]
PSEUDO_PERCENTS = [25, 50, 75, 100]


def seed_everything():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_dataset(dataset):
    root      = os.path.join("data", dataset)
    classes   = DATASET_CLASSES[dataset]
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    images, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(root, cls)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            images.append(transform(img).numpy())
            labels.append(idx)
    X = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    X, y = sk_shuffle(X, y, random_state=SEED)
    return X, y, classes


def get_folds(X, y, xu, n_splits=5, n_samples=1000):
    kf = KFold(n_splits=n_splits, shuffle=False)
    for fold_idx, (dev_idx, test_idx) in enumerate(kf.split(X)):
        X_dev, y_dev = X[dev_idx], y[dev_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        if xu:
            if n_samples is not None and len(X_dev) > n_samples:
                rng = np.random.default_rng(SEED)
                while True:
                    idx = rng.choice(len(X_dev), size=n_samples, replace=False)
                    if len(np.unique(y_dev[idx])) == len(np.unique(y)):
                        break
                X_dev, y_dev = X_dev[idx], y_dev[idx]
            while True:
                X_lbl, X_xu, y_lbl, y_xu = train_test_split(
                    X_dev, y_dev, train_size=0.50,
                    random_state=np.random.randint(0, 1_000_000)
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_lbl, y_lbl, test_size=0.15,
                    random_state=np.random.randint(0, 1_000_000)
                )
                if len(np.unique(y_train)) == len(np.unique(y)):
                    break
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_dev, y_dev, test_size=0.15, random_state=SEED
            )
            X_xu, y_xu = None, None
        mean = X_train.mean(axis=(0, 2, 3))
        std  = X_train.std(axis=(0, 2, 3))
        std  = np.where(std < 1e-6, 1.0, std)
        def norm(a): return (a - mean[None,:,None,None]) / std[None,:,None,None]
        yield fold_idx, {
            "train": (norm(X_train), y_train),
            "val":   (norm(X_val),   y_val),
            "xu":    (norm(X_xu), y_xu) if xu else (None, None),
            "test":  (norm(X_test),  y_test),
        }


def select_pseudolabels(probs, threshold, percentage):
    max_p  = probs.max(axis=1)
    pseudo = probs.argmax(axis=1)
    above  = np.where(max_p >= threshold)[0]
    if len(above) == 0:
        return np.zeros(len(probs), dtype=bool), pseudo
    n_use   = max(1, int(np.ceil(len(above) * percentage / 100.0)))
    ordered = above[np.argsort(max_p[above])[::-1]][:n_use]
    mask    = np.zeros(len(probs), dtype=bool)
    mask[ordered] = True
    return mask, pseudo


def hybrid_select(probs, xai_scores, alpha, threshold, percentage=100):
    max_probs     = probs.max(axis=1)
    pseudo_labels = probs.argmax(axis=1)
    scores        = (1.0 - alpha) * max_probs + alpha * xai_scores
    valid_idx     = np.where(scores >= threshold)[0]
    if len(valid_idx) == 0:
        return np.zeros(len(probs), dtype=bool), pseudo_labels
    n_keep   = max(1, int(np.ceil(len(valid_idx) * percentage / 100.0)))
    selected = valid_idx[np.argsort(scores[valid_idx])[::-1][:n_keep]]
    mask     = np.zeros(len(probs), dtype=bool)
    mask[selected] = True
    return mask, pseudo_labels


def _shap_importance(model, X):
    N, C, H, W = X.shape
    X_np = X.transpose(0, 2, 3, 1)
    def f(x):
        x_chw = x.transpose(0, 3, 1, 2).astype(np.float32)
        return model.predict_proba(x_chw)
    masker     = shap.maskers.Image(mask_value=0, shape=X_np[0].shape)
    explainer  = shap.Explainer(f, masker)
    probs      = model.predict_proba(X)
    preds      = probs.argmax(axis=1)
    shap_values = explainer(X_np)
    values     = shap_values.values
    importance = np.zeros((N, H * W), dtype=np.float32)
    for i in range(N):
        shap_map = values[i, :, :, :, preds[i]]
        pixel_importance = np.abs(shap_map).mean(axis=2)
        importance[i] = pixel_importance.flatten()
    return importance


def compute_xai_importance(model, X):
    return _shap_importance(model, X)


def compute_xai_scores(model, X):
    importance = compute_xai_importance(model, X)
    scores     = importance.mean(axis=1)
    mn, mx     = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + 1e-8)


def mask_from_importance(X, importance, level):
    if level == 0:
        return X.copy()
    N, C, H, W = X.shape
    Xm     = X.copy()
    n_mask = int(H * W * level / 100.0)
    for i in range(N):
        pix = np.argsort(importance[i])[::-1][:n_mask]
        Xm[i, :, pix // W, pix % W] = 0.0
    return Xm


def save_xai_visualization(X, importance, level, out_dir, dataset, model_name, fold):
    os.makedirs(out_dir, exist_ok=True)
    img     = X[0].copy()
    C, H, W = img.shape
    img     = img.transpose(1, 2, 0)
    img     = img - img.min()
    img     = img / (img.max() + 1e-8)
    if level > 0:
        n_mask     = int(H * W * level / 100.0)
        pix        = np.argsort(importance[0])[::-1][:n_mask]
        rows, cols = pix // W, pix % W
        img[rows, cols, 0] = 1.0
        img[rows, cols, 1] = 0.0
        img[rows, cols, 2] = 0.0
    plt.imsave(
        os.path.join(out_dir, f"{dataset}_k{fold}_shap{level}.png"),
        img
    )


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=2)
    print("saved:", path)


def out_path(out_dir, dataset, model, weights, experiment, fold,
             threshold=None, pseudo=None, alpha=None, xai_level=None):
    parts = [dataset, model, weights]
    if experiment == "full":
        parts.append("full")
    elif experiment == "threshold":
        parts.append(f"threshold_thr{str(int(threshold*100)).zfill(3)}")
    elif experiment == "pseudo":
        parts.append(f"pseudo{pseudo}_thr{str(int(threshold*100)).zfill(3)}")
    elif experiment == "xai":
        parts.append(f"xai{xai_level}")
    elif experiment == "hybrid":
        parts.append(f"hybrid_alpha{str(alpha).replace('.','')}")
    parts.append(f"k{fold+1}")
    return os.path.join(out_dir, "_".join(parts) + ".pkl")