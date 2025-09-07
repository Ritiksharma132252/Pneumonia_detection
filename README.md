# 🩺 Pneumonia Detection using Deep Learning

##  Project Overview
This project focuses on building a **deep learning-based model** to detect **pneumonia** from chest X-ray images.  
It leverages **Convolutional Neural Networks (CNNs)** and transfer learning (e.g., **VGG16**) to classify whether a patient has pneumonia or not.  

The goal is to assist healthcare professionals with a reliable tool for early diagnosis.

---

##  Dataset
- **Source:** [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- The dataset contains **chest X-ray images** categorized into:
  - `Normal`
  - `Pneumonia`

Dataset is split into:
- **Train:** Model training
- **Validation:** Hyperparameter tuning
- **Test:** Final evaluation

---

## Tech Stack
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras (VGG16 transfer learning)
- **Other Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
- **Environment:** Jupyter Notebook / VS Code

---

##  Project Workflow
1. **Data Preprocessing**
   - Image resizing & normalization
   - Data augmentation (rotation, flipping, zoom)
2. **Model Building**
   - Baseline CNN model
   - Transfer learning with VGG16
3. **Training**
   - Optimizer: Adam
   - Loss: Binary Crossentropy
   - Metrics: Accuracy
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix & ROC-AUC curve
5. **Prediction**
   - Test on unseen X-ray images

---

## Results
- Achieved **~94% accuracy** on the test set using **VGG16 transfer learning**.
- Outperformed baseline CNN in both accuracy and generalization.

---

## Sample Results
| Normal Case | Pneumonia Case |
|-------------|----------------|
| ![Normal](sample_images/normal.png) | ![Pneumonia](sample_images/pneumonia.png) |

---

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/ritiksharma132252/pneumonia_detection.git
   cd pneumonia-detection
