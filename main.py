#!/usr/bin/env python
import sys
import numpy as np
from sklearn.metrics import f1_score

from utils import (
    seed_everything, load_dataset, get_folds,
    select_pseudolabels, hybrid_select,
    compute_xai_importance, compute_xai_scores, mask_from_importance,
    save_xai_visualization, save_pkl, out_path,
    PSEUDO_PERCENTS, XAI_LEVELS, DATASET_CLASSES,
)

# python main.py <dataset> <min_fold> <max_fold> <model> <weights> <experiment>
#                [--out DIR]
#
# experiments: full | ssl | xai
#
# examples:
#   python main.py kidney 1 5 radimagenet medical full
#   python main.py kidney 1 5 radimagenet medical ssl
#   python main.py kidney 1 5 radimagenet medical xai
#   python main.py kidney 1 5 biomedclip  medical xai

DATASET    = sys.argv[1]
min_fold   = int(sys.argv[2])
max_fold   = int(sys.argv[3])
MODEL_NAME = sys.argv[4]
WEIGHTS    = sys.argv[5]
EXPERIMENT = sys.argv[6]

_args = sys.argv[7:]
def _get(flag, default):
    return type(default)(_args[_args.index(flag) + 1]) if flag in _args else default

OUT_DIR = _get("--out", "out")

N_CLASSES = len(DATASET_CLASSES[DATASET])
AVG       = "binary" if N_CLASSES == 2 else "weighted"


def make_model():
    if MODEL_NAME == "radimagenet":
        from RadImageNet import RadImageNetClassifier
        return RadImageNetClassifier(weights=WEIGHTS)
    elif MODEL_NAME == "siglip2":
        from SigLIP2 import SigLIP2Classifier
        return SigLIP2Classifier(weights=WEIGHTS)
    else:
        from BiomedCLIP import BiomedCLIPClassifier
        return BiomedCLIPClassifier(weights=WEIGHTS)


seed_everything()
print(DATASET, MODEL_NAME, WEIGHTS, EXPERIMENT)

X, y, classes = load_dataset(DATASET)
print("classes:", classes, "| N:", len(X))

XU_OPT = EXPERIMENT in ["ssl", "xai"]

for fold_idx, splits in get_folds(X, y, xu=XU_OPT, n_splits=5):
    k = fold_idx + 1
    if k not in range(min_fold, max_fold + 1):
        continue
    print(f"\nfold: {k}")

    (X_train, y_train) = splits["train"]
    (X_val,   y_val  ) = splits["val"]
    (X_xu,    y_xu   ) = splits["xu"]
    (X_test,  y_test ) = splits["test"]

    # Experiment 1: full
    if EXPERIMENT == "full":
        model = make_model()
        model.fit(X_train, y_train, X_val, y_val)
        probs = model.predict_proba(X_test)
        save_pkl({"y_true": y_test, "probs": probs},
                 out_path(OUT_DIR, DATASET, MODEL_NAME, WEIGHTS, "full", fold_idx))
        del model

    # Experiment 2: xai masking
    elif EXPERIMENT == "xai":
        model      = make_model()
        model.fit(X_train, y_train, X_val, y_val)
        importance = compute_xai_importance(model, X_test)

        probs_orig = model.predict_proba(X_test)
        save_pkl({"y_true": y_test, "probs": probs_orig},
                 out_path(OUT_DIR, DATASET, MODEL_NAME, WEIGHTS,
                          "xai", fold_idx, xai_level=0))
        save_xai_visualization(X_test, importance, 0,
                               OUT_DIR, DATASET, MODEL_NAME, k)

        for level in [l for l in XAI_LEVELS if l > 0]:
            print(f"  xai level={level}%")
            X_masked = mask_from_importance(X_test, importance, level)
            probs    = model.predict_proba(X_masked)
            save_pkl({"y_true": y_test, "probs": probs},
                     out_path(OUT_DIR, DATASET, MODEL_NAME, WEIGHTS,
                              "xai", fold_idx, xai_level=level))
            save_xai_visualization(X_test, importance, level,
                                   OUT_DIR, DATASET, MODEL_NAME, k)
        del model

    # Experiment 3: ssl (threshold ? pseudo ? hybrid)
    elif EXPERIMENT == "ssl":
        teacher  = make_model()
        teacher.fit(X_train, y_train, X_val, y_val)
        probs_xu = teacher.predict_proba(X_xu)

        xai_scores = compute_xai_scores(teacher, X_xu)
        for alpha in [0.0, 0.5, 1.0]:
            mask, pseudo_labels = hybrid_select(probs_xu, xai_scores, alpha, 0.90)
            n_acc = mask.sum()
            print(f"  alpha={alpha} accepted={n_acc}/{len(X_xu)}")

            if n_acc > 0:
                X_aug = np.concatenate([X_train, X_xu[mask]])
                y_aug = np.concatenate([y_train, pseudo_labels[mask]])
            else:
                X_aug, y_aug = X_train.copy(), y_train.copy()

            student = make_model()
            student.fit(X_aug, y_aug, X_val, y_val)
            probs   = student.predict_proba(X_test)
            save_pkl({"y_true": y_test,
                      "probs": probs,
                      "y_xu_true": y_xu,
                      "y_xu_pseudo": pseudo_labels,
                      "y_xu_probs": probs_xu,
                      "y_xai_scores": xai_scores},
                     out_path(OUT_DIR, DATASET, MODEL_NAME, WEIGHTS,
                              "hybrid", fold_idx, alpha=alpha))
            del student

        del teacher