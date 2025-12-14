# Explainable AI for Chest X-Ray Pneumonia Detection (DenseNet121 + Grad-CAM + LIME)

This repository contains the implementation for **Portfolio 2 – Sheet 2 (Assignment 1: Explainable AI for Medical Imaging)**.  
We fine-tune a **DenseNet121** classifier on the Kaggle Chest X-Ray Pneumonia dataset (NORMAL vs PNEUMONIA) and generate explanations using **Grad-CAM** and **LIME**.

---

## Repository Structure

project2/
├── models/
│ └── train_densenet_with_XAI.py # training + standalone XAI generation (Grad-CAM, LIME)
├── src/
│ ├── eval_metrics.py # evaluation metrics + saved figures (confusion matrix, ROC)
│ ├── xai_test_cases_v1.py # TEST-only: creates side-by-side (Original | Grad-CAM | LIME)
├── results/ # generated evaluation outputs (ROC, confusion matrix, txt)
├── xai_outputs_v1/ # side-by-side figures for TP/TN/FP/FN (presentation-ready)
├── project2part1pres.pptx
└── README.md


---

## Important Note About Missing Files (Dataset + Model)

### Dataset not included
The dataset directory is **not included in the GitHub repository** because it is too large.  
To run training/evaluation locally, download the Kaggle dataset and place it here:

data/chest_xray/chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── val/ (NOTE: not used in our pipeline; we create val via 80/20 split of train)


### Trained `.pth` model not included
The trained model file `best_densenet121_chestxray.pth` is also **not included** due to size constraints / repository cleanliness.  
If you want to reproduce results:
- run the training script to generate the `.pth`, or
- place your own trained checkpoint at:

best_densenet121_chestxray.pth


---

## Why XAI Outputs Are Included in the Repo

The **XAI visualizations are included** (`xai_outputs_v1/`) so that:
- explanations can be reviewed immediately without re-running heavy computations,
- the presentation figures are easy to access,
- everything (code + results + slides) is stored in one place for grading convenience.

---

## Setup

Create and activate your environment (example):


conda create -n torch_gpu python=3.10 -y
conda activate torch_gpu
pip install torch torchvision torchaudio
pip install numpy matplotlib pillow scikit-learn lime scikit-image

Running the Code
1) Training + bulk XAI generation

This script trains DenseNet121 (if a checkpoint does not exist) and generates Grad-CAM and LIME explanations:

python models/train_densenet_with_XAI.py

Outputs:

    checkpoint: best_densenet121_chestxray.pth

    Grad-CAM images: xai_outputs/gradcam/

    LIME images: xai_outputs/lime/

2) Evaluation metrics (test set)

Generates quantitative evaluation results and saves figures:

python src/eval_metrics.py

Outputs (saved to results/):

    confusion_matrix.png

    roc_curve.png

    classification_report.txt

    metrics_summary.txt

3) Presentation-ready XAI triplets from TEST set (TP/TN/FP/FN)

Creates side-by-side figures using only the test set:

    Original image

    Grad-CAM overlay

    LIME explanation

It selects up to 2 examples per case:

    TP: true pneumonia, predicted pneumonia

    TN: true normal, predicted normal

    FP: true normal, predicted pneumonia

    FN: true pneumonia, predicted normal

Run:

python src/xai_test_cases_v1.py

Outputs:

    xai_outputs_v1/TP_example_00_v1.png, TN_example_00_v1.png, etc.

4) Dataset size & class distribution

Counts images in train/test/(original val) and reports the effective 80/20 split:

python src/count_dataset_images.py
python src/count_dataset_images_detailed.py

Notes on Experimental Setup

    Model: DenseNet121 (ImageNet initialization) + custom classifier head

    Preprocessing: grayscale → resize 224×224 → repeat to 3 channels → normalize

    Train augmentation: random horizontal flip

    Train/Val split: validation is created by random 80/20 split from train/ (original val/ folder is not used)

    Class imbalance: handled using WeightedRandomSampler in the training dataloader

    XAI methods: Grad-CAM (gradient-based) and LIME (perturbation + superpixels)
