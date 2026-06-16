# Skin Lesion Classification using GAN-Based Data Balancing

## Overview
This project focuses on **skin lesion classification** using deep learning on the HAM10000 dataset, addressing the critical issue of **class imbalance**.

We compare three approaches:
- **A:** Baseline CNN (no augmentation)  
- **B:** CNN with standard augmentation  
- **C:** CNN trained on a **GAN-balanced dataset**

The GAN-based approach significantly improves performance, especially for rare and clinically important classes.

---

## Problem Statement
The HAM10000 dataset is **highly imbalanced**:
- Majority class dominates (Melanocytic Nevi)
- Rare classes (e.g., Dermatofibroma, Melanoma) have very few samples

This causes:
- Biased predictions
- Poor recall on critical classes
- Misleading accuracy

---

## Proposed Solution
We use **per-class DCGANs** to generate synthetic images and fully balance the dataset.

### Key Idea
- Learn class distributions using GANs  
- Generate **new synthetic samples**  
- Create a **fully balanced dataset**

---

## Model Architecture

### Classifier: ResNet-18 (Transfer Learning)
- Pretrained on ImageNet  
- Modified classification head:
  ```
  Dropout(0.4)
  → FC(512 → 256)
  → ReLU
  → FC(256 → 7)
  ```

### Training Strategy
- **Phase 1:** Freeze backbone, train head  
- **Phase 2:** Fine-tune entire network  

---

### GAN Model: DCGAN
- One GAN per class  
- Input: 100-dim noise vector  
- Output: 64×64 RGB image  

---

## Project Structure

```
├── 01_eda.ipynb
├── 02_baseline_cnn.ipynb
├── 03_dcgan_allclass.ipynb
├── 04_comparison.ipynb
├── Report.pdf
├── data/
├── gan_generated/
└── models/
```

---

## Experiments

### Experiment A — Baseline CNN
- No augmentation  
- Uses weighted loss  

**Macro F1:** 0.4793  

---

### Experiment B — Standard Augmentation
- Flip, rotation, color jitter, affine transforms  

**Macro F1:** 0.4733  

---

### Experiment C — GAN-Based Balancing
- Fully balanced dataset  

**Macro F1:** **0.6680**

---

## Results Summary

| Experiment | Macro F1 |
|-----------|---------|
| Baseline (A) | 0.4793 |
| Augmentation (B) | 0.4733 |
| GAN Balanced (C) | **0.6680** |

---

### Key Improvements
- Dermatofibroma: 0.385 → 0.625  
- Melanoma: 0.283 → 0.514  
- All classes improved  

---

## Grad-CAM Visualization

We use **Grad-CAM** to interpret model predictions.

### Purpose
- Visualize model attention  
- Verify meaningful feature learning  

### Observations
- Baseline → scattered attention  
- GAN model → focused on lesion  

---

## Techniques Used
- Transfer Learning (ResNet-18)  
- GAN-based Data Augmentation (DCGAN)  
- Weighted Sampling  
- Class-weighted Loss  
- Grad-CAM  

---

## Evaluation Metrics
- Precision  
- Recall  
- F1-score  
- **Macro F1 (main metric)**  

---

## How to Run

### 1. Clone repository
```bash
git clone <repo-link>
cd project-folder
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks
```bash
01_eda.ipynb
02_baseline_cnn.ipynb
03_dcgan_allclass.ipynb
04_comparison.ipynb
```
GAN-based data balancing:
- Solves extreme class imbalance  
- Improves classification performance  
- Enhances model interpretability  

👉 Makes the system more reliable for medical use
