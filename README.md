# Chronic Kidney Disease Detection using Machine Learning

## Overview

Chronic Kidney Disease (CKD) is a serious health condition that affects millions of people worldwide. Early detection of CKD can help prevent kidney failure and improve patient outcomes.

This project uses Machine Learning techniques to predict whether a patient is likely to have Chronic Kidney Disease based on various medical parameters. The system analyzes patient health data and provides predictions that can assist healthcare professionals in early diagnosis and decision-making.

---

## Features

* Data preprocessing and cleaning
* Handling missing values
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine Learning model training and evaluation
* CKD prediction based on patient medical data
* Performance comparison of different algorithms
* User-friendly prediction workflow

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## Dataset

The dataset contains various medical attributes related to kidney health, including:

* Age
* Blood Pressure
* Specific Gravity
* Albumin
* Sugar
* Red Blood Cells
* Pus Cell
* Blood Glucose Random
* Blood Urea
* Serum Creatinine
* Sodium
* Potassium
* Hemoglobin
* Packed Cell Volume
* White Blood Cell Count
* Red Blood Cell Count

Target Variable:

* CKD (Chronic Kidney Disease)
* Not CKD

Dataset Source:
UCI Machine Learning Repository – Chronic Kidney Disease Dataset

---

## Machine Learning Workflow

### 1. Data Collection

Medical data is collected from the CKD dataset.

### 2. Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Data cleaning

### 3. Exploratory Data Analysis

* Distribution analysis
* Correlation analysis
* Feature importance evaluation

### 4. Model Training

Multiple Machine Learning models can be trained and compared.

Examples:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

### 5. Model Evaluation

Performance metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Project Structure

```text
Chronic-Kidney-Disease-Detection/
│
├── dataset/
├── notebooks/
├── models/
├── images/
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rohangowda272/Chronic-Kidney-Disease-Detection.git
```

Move into the project directory:

```bash
cd Chronic-Kidney-Disease-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python app.py
```

---

## Results

The trained model successfully predicts the likelihood of Chronic Kidney Disease based on patient health parameters.

Key benefits:

* Early disease detection
* Faster screening process
* Data-driven healthcare support
* Improved diagnostic assistance

---

## Future Improvements

* Deep Learning implementation
* Web-based deployment
* Real-time patient data integration
* Mobile application support
* Explainable AI (XAI) integration
* Cloud deployment

---

## Author

Rohan M

LinkedIn:
https://www.linkedin.com/in/rohan-m-3605991a7/

GitHub:
https://github.com/rohangowda272

---

## Disclaimer

This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
