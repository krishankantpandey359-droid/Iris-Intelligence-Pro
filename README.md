# 🌸 Iris Intelligence Pro

### Premium Machine Learning Classification & Interactive Analytics Dashboard

An end-to-end Machine Learning project built with Python and Scikit-learn to classify Iris flower species. The project combines data analysis, model training, evaluation, and an interactive Streamlit dashboard with a modern analytics interface.

---

## 📸 Dashboard Showcase

![Iris Intelligence Pro Dashboard](screenshots/iris-intelligence-pro-showcase.png)

**Dashboard Modules**
- Executive Overview
- Data Explorer
- Predict Studio
- Batch Prediction
- Model Intelligence
- About Project

---

## 📌 Project Overview

Iris Intelligence Pro is a multiclass classification project that predicts Iris flower species using four numerical measurements.

The project uses the classic Iris dataset, containing 150 observations, four input features, and three target classes:

- Setosa
- Versicolor
- Virginica

It follows an end-to-end Machine Learning workflow, including data validation, Exploratory Data Analysis (EDA), model comparison, hyperparameter tuning, cross-validation, model evaluation, and interactive predictions.

## ✨ Key Features

- Dataset loading and validation
- Exploratory Data Analysis (EDA)
- Feature distribution and correlation analysis
- Five Machine Learning classification algorithms
- Hyperparameter tuning using GridSearchCV
- Stratified 5-fold cross-validation
- Model selection using cross-validation accuracy
- Held-out test evaluation
- Confusion matrix and classification report
- ROC curve analysis
- Permutation feature importance
- Interactive Streamlit dashboard
- Single flower prediction
- CSV-based batch prediction
- Prediction confidence display
- Automated data validation tests

## 🧠 Machine Learning Models

Five classification algorithms are evaluated:

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification |
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Support Vector Machine (SVM) | Margin-based classification |
| Random Forest | Ensemble learning |
| Gradient Boosting | Boosting-based classification |

The best model is selected using cross-validation accuracy on the training dataset.

## 📊 Model Performance

Results from the recorded training run:

| Metric | SVM Result |
|---|---:|
| Cross-Validation Accuracy | 97.50% |
| Test Accuracy | 93.33% |
| Macro Precision | 93.33% |
| Macro Recall | 93.33% |
| Macro F1-Score | 93.33% |

**Selected Model:** Support Vector Machine (SVM)

These results are based on a stratified 80:20 train-test split with random state 42 and stratified 5-fold cross-validation.

The test set contains 30 observations. One incorrect prediction changes test accuracy by approximately 3.33 percentage points.

## 📈 Dashboard Modules

### 1. Executive Overview
- Total dataset records
- Number of features and species
- Missing value status
- Species distribution
- Dataset statistical summary

### 2. Data Explorer
- Feature distribution analysis
- Histogram, box plot, and violin plot
- Feature relationship analysis
- Correlation heatmap

### 3. Predict Studio
- Interactive flower measurement inputs
- Single flower species prediction
- Prediction confidence
- Class probability visualization
- Prediction history

### 4. Batch Prediction
- CSV file upload
- Required feature validation
- Multiple flower predictions
- Prediction distribution
- Downloadable prediction results

### 5. Model Intelligence
- Model comparison table
- Cross-validation accuracy chart
- Confusion matrix
- ROC curve analysis
- Feature importance
- Classification report
- Training metadata

### 6. About Project
- Project objective
- Dataset information
- Machine Learning algorithms
- Methodology
- Technology stack
- Model performance

## 🗂️ Project Structure

```text
Iris-Intelligence-Pro/
│
├── app.py
├── app_backup.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── iris.csv
│
├── models/
│   └── best_iris_classifier.joblib
│
├── notebooks/
│   └── Task1_Iris_Flower_Classification.ipynb
│
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   ├── roc_curve_data.csv
│   ├── run_metadata.json
│   └── test_predictions.csv
│
├── reports/
│   └── Project_Report.md
│
├── screenshots/
│   └── iris-intelligence-pro-showcase.png
│
├── src/
│   ├── data_utils.py
│   └── train.py
│
└── tests/
    └── test_project.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/krishankantpandey359-droid/Iris-Intelligence-Pro.git
cd Iris-Intelligence-Pro
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the Models

```bash
python src/train.py
```

### 5. Run Automated Tests

```bash
python -m pytest -q
```

### 6. Launch the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 📁 Dataset Information

The project uses the classic Iris flower dataset.

| Dataset Property | Value |
|---|---:|
| Total Observations | 150 |
| Input Features | 4 |
| Target Classes | 3 |
| Missing Values | 0 |

**Input Features**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

All measurements are recorded in centimetres.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Development | Jupyter Notebook |
| Testing | Pytest |

## 🔍 Methodology

1. Load and validate the Iris dataset.
2. Perform Exploratory Data Analysis.
3. Split the dataset into stratified training and test sets.
4. Tune five classification models using GridSearchCV.
5. Apply stratified 5-fold cross-validation.
6. Select the best model using cross-validation accuracy.
7. Evaluate the selected model on the held-out test set.
8. Save the trained model and evaluation outputs.
9. Use the trained model for interactive predictions.

Feature scaling is performed inside pipelines for scale-sensitive models. This prevents preprocessing leakage across cross-validation folds.

## 🎯 Learning Outcomes

This project provided practical experience in:

- End-to-end Machine Learning workflows
- Multiclass classification
- Data preprocessing and validation
- Exploratory Data Analysis
- Model comparison and hyperparameter tuning
- Cross-validation and model evaluation
- Data visualization
- Streamlit dashboard development
- Python project organization
- Automated testing

## ⚠️ Limitations

- The Iris dataset is a small benchmark dataset.
- The test set contains only 30 observations.
- High benchmark performance does not guarantee similar performance on new datasets.
- Prediction confidence is not the same as guaranteed prediction accuracy.
- Model performance depends on the dataset and evaluation setup.

## 👩‍💻 Author

**Srishti Pandey**

MCA Data Science Student

GitHub: [krishankantpandey359-droid](https://github.com/krishankantpandey359-droid)

**Project Repository:**  
[Iris Intelligence Pro](https://github.com/krishankantpandey359-droid/Iris-Intelligence-Pro)

---

⭐ Thank you for exploring Iris Intelligence Pro!
