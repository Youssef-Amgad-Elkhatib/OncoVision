# 🧠 OncoVision

OncoVision is a deep learning project for **brain tumor MRI classification** using Convolutional Neural Networks (CNNs).  
It focuses on detecting and classifying brain tumors into categories such as **Glioma, Meningioma, Pituitary, and No Tumor**.  

The project explores data preprocessing, augmentation, model design, training with checkpoints, and evaluation through metrics and visualizations.

---
# 🎥 Video
[![OncoVision](https://ytcards.demolab.com/?id=wJ710_eJ5uw&title=OncoVision&lang=en&timestamp=1756150574&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=2&width=300&border_radius=5&duration=166 "OncoVision")](https://www.youtube.com/watch?v=wJ710_eJ5uw)

---

## 📌 Features
- 📂 **Organized Jupyter Notebook** with clean sections and icons for readability  
- 🖼️ **Data Augmentation** (flips, rotations, zooms, contrast) to improve generalization  
- 🏗️ **Custom CNN Architecture** tailored for MRI image classification  
- 🎯 **Dual Metric Checkpointing** – saves best model based on validation accuracy & loss  
- 📊 **Comprehensive Evaluation** with:
  - Accuracy & loss curves  
  - Confusion matrix  
  - Classification report  
  - Sample predictions with visualizations  
- ⚡ **High Performance** – achieved **99.3% test accuracy** on held-out MRI images  

---

## 📂 Dataset
This project uses the **Brain Tumor MRI Dataset** containing four classes:
- 🧩 Glioma  
- 🧩 Meningioma  
- 🧩 Pituitary  
- 🚫 No Tumor  

> The dataset is available on [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **TensorFlow / Keras** (deep learning)
- **scikit-learn** (metrics & data split)
- **Matplotlib / Seaborn** (visualizations)
- **NumPy / Pandas** (data handling)

---
