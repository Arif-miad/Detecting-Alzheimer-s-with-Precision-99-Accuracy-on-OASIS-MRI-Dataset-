
<div style="text-align: center;">
  <h3>⭐ 🧠 Alzheimer’s Disease Detection with OASIS MRI Dataset⭐</h3>
</div>
Welcome to the repository for **Alzheimer’s Disease Detection using the OASIS MRI Dataset**! This project demonstrates a complete end-to-end pipeline for classifying brain MRI scans into four categories: **Non-Demented**, **Very Mild Demented**, **Mild Demented**, and **Demented**, achieving exceptional performance with deep learning techniques. Below, you will find a detailed explanation of the dataset, methodologies, results, and how you can explore and contribute to the project.

---

## 🌟 **Key Highlights**

- **Dataset:** OASIS MRI dataset with 80,000 brain MRI images in .jpg format, sourced from .nii files.
- **Classes:** Non-Demented, Very Mild Demented, Mild Demented, Demented.
- **Model Achievements:** Achieved **99% accuracy** with optimized CNN architectures.
- **End-to-End Workflow:** Includes data preprocessing, augmentation, model training, evaluation, and visualization.
- **Applications:** Early detection and progression analysis of Alzheimer’s disease using MRI scans.

---

## 📂 **Dataset Overview**

The dataset used in this project is sourced from the [OASIS](https://sites.wustl.edu/oasisbrains/) initiative. It contains MRI scans converted to .jpg format for convenience. Here's an in-depth view:

- **Source:** OASIS-1 Cross-Sectional Dataset.
- **Format Conversion:** Original .img and .hdr files were converted to .nii format using FSL (FMRIB Software Library) and further processed into .jpg files.
- **Total Size:** 1.3 GB.
- **Number of Patients:** 461.
- **Slices Selected:** Slices along the z-axis from 100 to 160.
- **Categories:**
  - Non-Demented
  - Very Mild Demented
  - Mild Demented
  - Demented

---

## 📊 **Workflow Explanation**

This repository provides a structured pipeline for working with the OASIS dataset. Below are the steps broken down:

### 🔍 **1. Data Collection and Preparation**

1. **Download the Dataset:**
   The dataset is hosted on a GitHub repository and segmented for easier handling.

2. **Data Organization:**
   - Split into training, validation, and testing datasets.
   - Maintain an 80:10:10 ratio for model evaluation.

3. **Preprocessing:**
   - Convert .nii files into 2D slices.
   - Normalize pixel intensity for optimal input.
   - Resize images to \(224 \times 224\) pixels for compatibility with deep learning models.

### 🛠️ **2. Data Augmentation**

To enhance the model's robustness, the following augmentations were applied:
- Rotation
- Horizontal flipping
- Zoom
- Brightness adjustments
- Random cropping

### 🤖 **3. Model Development**

1. **Baseline Model:**
   - Built a Convolutional Neural Network (CNN) with three convolutional layers, ReLU activations, and max-pooling.

2. **Advanced Models:**
   - Transfer learning with **pretrained models** like VGG16, ResNet50, and InceptionV3.
   - Fine-tuned layers to optimize performance.

3. **Training:**
   - Batch size: 32.
   - Epochs: 50.
   - Optimizer: Adam.
   - Loss function: Categorical Crossentropy.

### 📈 **4. Evaluation**

- Metrics Used:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Confusion Matrix and Classification Reports for detailed insights.
- Achieved an accuracy of **99%** on the test set.

### 📊 **5. Visualization**

1. **Data Visualization:**
   - Displayed MRI slices and class distributions.

2. **Model Visualization:**
   - Plotted loss and accuracy curves for training and validation.

3. **Heatmaps:**
   - Used Grad-CAM to highlight regions contributing to predictions.

---

## 💡 **How to Use This Repository**

### 🔧 **Setup Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/oasis-alzheimers-detection.git
   cd oasis-alzheimers-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Follow instructions to download the OASIS dataset.
   - Place the data in the `data/` directory.

4. Run the training script:
   ```bash
   python train_model.py
   ```

5. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```

---

## 🌟 **Features of This Repository**

- **Jupyter Notebooks:** Interactive walkthroughs for each step of the workflow.
- **Scripts:** Modular Python scripts for preprocessing, training, and evaluation.
- **Pretrained Models:** Downloadable weights for immediate use.
- **Visualization Tools:** Notebooks to create heatmaps and interpret model decisions.

---

## 📌 **Applications and Future Work**

1. **Clinical Applications:**
   - Aid in early diagnosis of Alzheimer’s.
   - Study progression stages for timely interventions.

2. **Research Opportunities:**
   - Test novel deep learning architectures.
   - Extend dataset with more MRI scans.

3. **Advanced Techniques:**
   - Apply 3D CNNs for volumetric analysis.
   - Explore attention-based models for enhanced performance.

---

## 🤝 **Contributions**

Contributions are welcome! Please follow the guidelines in `CONTRIBUTING.md` to submit your ideas or improvements.

---

## 🛠️ **Technologies Used**

- Python 🐍
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pandas

---

## 🔗 **Connect with Me**

- **Kaggle Profile:** [Arif Mia on Kaggle](https://www.kaggle.com/code/arifmia/detecting-alzheimer-s-with-precision-99-accur)
- **LinkedIn Profile:** [Arif Mia on LinkedIn](www.linkedin.com/in/arif-miah-8751bb217)
- **Email:** arifmiahcse@gmail.com

---

## 📜 **Acknowledgments**

- **OASIS Dataset:** Gratitude to the researchers and institutions for providing this invaluable dataset.
- **Community:** Thanks to the deep learning community for inspiration and support.

---

## 📄 **License**

This project is licensed under the MIT License. See `LICENSE` for details.

---

<div style="text-align: center;">
  <h3>⭐ If you find this repository helpful, please give it a star! ⭐</h3>
</div>

