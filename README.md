# SVM Diabetes Classifier

This project implements a **Support Vector Machine (SVM)** classifier to detect diabetes in female patients of Pima Indian heritage, using the well-known **Pima Indians Diabetes Database**. The model achieves an accuracy of **~81%** on the test set.  

---

## Dataset
The dataset used is `diabetes.csv`, which contains the following features:  

- **Pregnancies**: Number of pregnancies  
- **Glucose**: Plasma glucose concentration  
- **BloodPressure**: Diastolic blood pressure (mm Hg)  
- **SkinThickness**: Triceps skinfold thickness (mm)  
- **Insulin**: 2-Hour serum insulin (mu U/ml)  
- **BMI**: Body Mass Index (weight in kg/(height in m)^2)  
- **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history  
- **Age**: Age in years  
- **Outcome**: Target variable (1 = diabetes, 0 = no diabetes)  

---

## Data Preprocessing
1. **Handling missing/zero values**  
   - Replaced zeros in `Glucose`, `BloodPressure`, and `SkinThickness` with the **median** of the column.  

2. **Outlier treatment**  
   - Used **Pearson’s median skewness coefficient** to check for skewness.  
   - If skewed, applied **Interquartile Range (IQR)** clipping.  
   - If normally distributed, applied **Z-score** clipping.  

3. **Train-Test Split**  
   - Split the dataset into **85% training** and **15% testing** using `train_test_split` with `random_state=0`.  

---

## Model
- Implemented an **SVM classifier** using the **Radial Basis Function (RBF) kernel**.  
- Trained on preprocessed training data.  
- Achieved **accuracy: 0.8103 (~81%)** on the test set.  

---

## Installation
Clone the repository and install required dependencies:  

```bash
git clone https://github.com/vedachari/Diabetes_SVM.git
cd Diabetes_SVM
pip install -r requirements.txt
