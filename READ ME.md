#  Brain Tumor Classification using Custom CNN

This project focuses on detecting and classifying brain tumors from MRI images using a custom-built Convolutional Neural Network (CNN). It identifies four categories of brain conditions:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

---

##  Project Overview

- The model was trained on MRI scan images divided into train, validation, and test sets.
- The classification task is multi-class (4 categories).
- Both a **Custom CNN** and a **Transfer Learning model** were tested and compared.

---

##  Model Performance Comparison

| Metric     | Custom CNN | Transfer Learning |
|------------|------------|-------------------|
| Accuracy   | 80.89%     | 41.06%            |
| Precision  | 81.27%     | 37.10%            |
| Recall     | 80.89%     | 41.06%            |
| F1-Score   | 80.80%     | 36.39%            |

 **Final Model Chosen:** Custom CNN  
 **Why:** The custom CNN significantly outperformed the transfer learning model across all key evaluation metrics.

---

##  Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## Project Structure

BrainTumorClassification/
├── custom_cnn_model.ipynb # Main notebook with training and evaluation
├── best_model_cnn.h5 # Final trained Keras model
├── test_samples/ # Sample images for prediction
└── README.md # Project documentation


##  How to Run the Project

1. Clone the repository:
https://github.com/Rounak36/Brain-Tumor-MRI-Image-Classification
markdown


2. Open `custom_cnn_model.ipynb` in Jupyter or Google Colab and run the cells.

3. Load the `.h5` model and make predictions on new MRI images.

---

##  Future Scope

- Integrate with a web interface using Streamlit
- Improve accuracy via data augmentation or more CNN layers
- Use Grad-CAM for tumor region visualization

---


##  Contact

Made with ** Rounak Saha**  
For queries or collaboration: [saharounak65@gmail.com]
