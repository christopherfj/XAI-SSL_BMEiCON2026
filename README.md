# Combining Explainability and Confidence for Pseudo-Label Selection in Biomedical Image Classification

This repository contains the source code accompanying the paper:

> **Combining Explainability and Confidence for Pseudo-Label Selection in Biomedical Image Classification**

## Overview

Biomedical image classification often suffers from the limited availability of expert annotations. This project investigates whether explainability can improve pseudo-label selection in semi-supervised learning by complementing traditional confidence-based selection.

The proposed framework combines prediction confidence with SHAP-derived explainability to rank unlabeled samples before pseudo-labeling.

Three vision foundation models are evaluated:

- **RadImageNet**
- **BiomedCLIP**
- **SigLIP2**

Experiments are performed on two biomedical image classification datasets:

- Kidney CT
- COVID-19 Chest X-ray

---

## Repository Structure

```
.
├── data/                  # Datasets
├── out/                   # Experimental outputs
├── pretrained/            # Pre-trained weights for RadImageNet (medical)
├── main.py                # Main experimental pipeline
├── utils.py               # Utility functions
├── RadImageNet.py         # RadImageNet classifier
├── BiomedCLIP.py          # BiomedCLIP classifier
├── SigLIP2.py             # SigLIP2 classifier
├── analyze_results.py     # Result analysis and figure generation
└── README.md
```

---

## Experimental Protocol

### Experiment 1 – Supervised Classification

Comparison of pretrained models under fully supervised training using standard classification metrics.

---

### Experiment 2 – Explainability-Based Masking

SHAP is used to identify the most relevant image regions for the predicted class. Increasing percentages of the most relevant regions are progressively masked to evaluate the robustness of the learned representations.

Masking levels:

- 0%
- 10%
- 25%
- 50%

---

### Experiment 3 – Semi-Supervised Learning

Pseudo-label selection is performed using a hybrid score that combines classifier confidence and SHAP-based explainability:

\[
\text{Score}=(1-\alpha)\times\text{Confidence}+\alpha\times\text{SHAP}
\]

Three values of the weighting parameter are evaluated:

- **α = 0.0** (confidence only)
- **α = 0.5** (hybrid)
- **α = 1.0** (SHAP only)

---

## Results

The repository generates:

- Supervised classification results
- Explainability masking analysis
- Semi-supervised learning performance
- Pseudo-label quality analysis
- Correlation between prediction uncertainty and SHAP scores
- Publication-ready figures and tables

All outputs are automatically stored in the `out/`, `Figures/`, and `Tables/` directories.

---

## Citation

If you use this repository in your research, please cite:

```text
Combining Explainability and Confidence for Pseudo-Label Selection in Biomedical Image Classification.
```

## License

This repository is intended for research and academic use.
