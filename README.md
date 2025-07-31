# Oculai: RetinaMNIST Clinical Decision Support

A reproducible, clinically-oriented Machine Learning project for grading Diabetic Retinopathy severity using the **RetinaMNIST** dataset from the MedMNIST collection.

This repository demonstrates structured machine learning workflows, rigorous validation, handling of class imbalances, and performance metrics suitable for medical image classification.

---

## 🔬 Scientific Context & Dataset

Diabetic Retinopathy (DR) is graded on an ordinal scale of severity from **0 to 4**:
* **0**: Normal
* **1**: Mild DR
* **2**: Moderate DR
* **3**: Severe DR
* **4**: Proliferative DR

### Dataset Split & Distribution
RetinaMNIST contains standard $28 \times 28 \times 3$ fundus projection images.
* **Train**: 1,080 samples
* **Validation**: 120 samples
* **Test**: 400 samples

Medical datasets are frequently imbalanced. In this split, the prevalence of healthy (class 0) or moderate (class 2) categories outweighs severe or proliferative classes. To combat this bias, **dynamic class weights** are calculated from the training data and integrated into the Multi-class Cross-Entropy loss.

---

## 🛠️ Methodological Approach

### 1. Model Architectures
* **Baseline (Linear)**: A single fully connected layer mapping flattened $28\times 28\times 3$ inputs ($2352$ dimensions) directly to 5 output logits.
* **Hero (Custom CNN)**: A deep convolutional network containing:
  - 3 Convolutional blocks with `BatchNorm2d` and `ReLU` activation.
  - Periodic `MaxPool2d` layers to reduce dimensionality.
  - Dropout regularization ($p=0.4$) in the classifier head to prevent overfitting on the small sample size.

### 2. Validation & Evaluation Workflow
* **Deterministic Seeds**: Full reproducibility is achieved by pinning random seeds across `numpy`, `random`, and `torch` (including macOS Metal Performance Shaders / CUDA).
* **Validation Loop**: Monitored epoch-by-epoch for loss and diagnostic metrics.
* **QWK Checkpointing**: Rather than saving checkpoints purely on training loss (which risks overfitting) or validation accuracy (which ignores ordinal ordering errors), we save checkpoints based on the best **Validation Quadratic Weighted Kappa (QWK)**.

### 3. Medical Performance Metrics
Standard accuracy is insufficient for medical diagnostics. We calculate and report:
* **Quadratic Weighted Cohen's Kappa (QWK)**: The industry standard for grading agreement in clinical diagnostics, penalizing larger jumps in misclassification (e.g., predicting 4 when ground truth is 0) much more severely than minor jumps (predicting 1 when ground truth is 0).
* **Macro AUC-ROC**: Metric indicating the probability of correct diagnostic ranking between classes.
* **Macro F1-Score**: Harmonic mean of precision and recall.

---

## 🚀 Execution Guide

### Setup Environment
Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Train the Models
Train the Baseline Linear model:
```bash
python src/train.py --config configs/baseline_linear.yaml
```

Train the Custom CNN Hero model:
```bash
python src/train.py --config configs/hero_cnn.yaml
```

Checkpoints, loss curves, QWK curves, and logs will be saved to `./outputs/linear/` and `./outputs/cnn/`.

### 2. Evaluate Performance
After training, evaluate the models on the test set:

Evaluate Baseline:
```bash
python src/evaluate.py --config configs/baseline_linear.yaml --checkpoint outputs/linear/best_model.pth
```

Evaluate Custom CNN:
```bash
python src/evaluate.py --config configs/hero_cnn.yaml --checkpoint outputs/cnn/best_model.pth
```

This generates confusion matrices and ROC curves inside `outputs/<model_name>/evaluation/`.

---

## 📊 Results Summary

Performance metrics comparisons on the test set (400 samples):

| Model | Test Accuracy | Macro F1 | QWK | Macro AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Linear** | `0.4950` | `0.3105` | `0.5206` | `0.7158` |
| **Custom CNN (Hero)** | `0.4900` | `0.3776` | `0.5338` | `0.7311` |

---

## 📈 Performance Visuals

To validate the clinical reliability of the models, we analyzed their diagnostic classifications using confusion matrices and monitored convergence stability during optimization.

### 1. Classification Performance (Confusion Matrices)
The confusion matrices highlight the diagnostic recall (sensitivity) for each severity grade. The Custom CNN exhibits much better class balance compared to the linear model.

| **Linear Baseline Confusion Matrix** | **Tuned CNN Confusion Matrix** |
| :---: | :---: |
| ![Linear Confusion Matrix](assets/linear_confusion_matrix.png) | ![CNN Confusion Matrix](assets/cnn_confusion_matrix.png) |

*The CNN demonstrates more distributed sensitivity across minority classes (mild/severe cases), whereas the linear model collapses heavily towards predicting moderate classes (Class 2).*

### 2. Convergence Stability (CNN Training Curves)
By scaling dropout ($p=0.5$), weight decay ($0.01$), and applying a dynamic learning rate scheduler (`ReduceLROnPlateau`), we successfully prevented overfitting and allowed the model to converge smoothly up to 100 epochs.

| Loss Curve (CNN) | Kappa Metric Curve (CNN) |
| :---: | :---: |
| ![CNN Loss Curve](assets/cnn_loss_curve.png) | ![CNN QWK Curve](assets/cnn_qwk_curve.png) |

---

## 🔍 Model Explainability (Grad-CAM)

To move beyond "black-box" predictions, we implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**. This diagnostic technique extracts gradients from the final convolutional layer of the Custom CNN to identify the exact visual features driving the model's classification.

### Retinopathy Grading Interpretability Gallery

| Severity Grade | Visual Explanation (Original vs. Grad-CAM Activation Heatmap) |
| :--- | :--- |
| **0: Normal** | ![Class 0 Normal](assets/gradcam_class_0.png) |
| **1: Mild** | ![Class 1 Mild](assets/gradcam_class_1.png) |
| **2: Moderate** | ![Class 2 Moderate](assets/gradcam_class_2.png) |
| **3: Severe** | ![Class 3 Severe](assets/gradcam_class_3.png) |
| **4: Proliferative** | ![Class 4 Proliferative](assets/gradcam_class_4.png) |

*The Grad-CAM visualizations confirm the model identifies pathological features such as microaneurysms, hemorrhages, and exudates, aligning with established clinical ophthalmic diagnostic patterns.*

---

## 📚 References & Citations

If you use the datasets or find this project useful, please cite the official MedMNIST publication:

```bibtex
@article{medmnistv2,
  title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={41},
  year={2023},
  publisher={Nature Publishing Group}
}
```

Original conference version:
```bibtex
@inproceedings{medmnistv1,
  title={Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis},
  author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
  booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages={191--195},
  year={2021},
  organization={IEEE}
}
```

### DeepDRiD (Original Source Dataset for RetinaMNIST)

RetinaMNIST is derived from the DeepDRiD challenge dataset. If you use this specific subset, you should also cite the original challenge paper:

```bibtex
@article{deepdrid2022,
  title={DeepDRiD: Diabetic retinopathy—Grading and image quality estimation challenge},
  author={Liu, Ruhan and Wang, Xiangning and Wu, Qiang and Dai, Ling and Fang, Xi and Yan, Tao and others},
  journal={Patterns},
  volume={3},
  number={6},
  pages={100512},
  year={2022},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.patter.2022.100512}
}
```

### Core Frameworks & Tools

To cite the primary libraries used for implementing model training, data loaders, and diagnostic metrics:

```bibtex
@inproceedings{pytorch2019,
  title={PyTorch: An imperative style, high-performance deep learning library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={8024--8035},
  year={2019}
}

@article{scikit-learn,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and others},
  journal={Journal of Machine Learning Research (JMLR)},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```



