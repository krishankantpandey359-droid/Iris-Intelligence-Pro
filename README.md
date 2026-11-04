# 🌸 Iris Intelligence Pro

### Machine Learning Classification & Interactive Analytics Dashboard

An end-to-end Machine Learning project that classifies Iris flower species using five classification algorithms and presents model insights through an interactive Streamlit dashboard.

## 📌 Project Overview

Iris Intelligence Pro is a multiclass classification project built using Python and Scikit-learn.

The project uses four flower measurements to predict one of three Iris species:

- Setosa
- Versicolor
- Virginica

It combines data preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, model evaluation, and interactive predictions.

## ✨ Key Features

- Data loading and validation
- Exploratory Data Analysis (EDA)
- Five Machine Learning classification algorithms
- Hyperparameter tuning using GridSearchCV
- Stratified 5-fold cross-validation
- Model evaluation on a separate test dataset
- Confusion matrix and classification report
- ROC curve analysis
- Permutation feature importance
- Interactive Streamlit dashboard
- Single and batch predictions
- Automated project tests

## 🧠 Machine Learning Models

The following algorithms are evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. Gradient Boosting

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

Selected Model: Support Vector Machine (SVM)

Note: Results are based on a single train-test split with random state 42. The Iris dataset contains 150 observations, so performance should be interpreted in the context of this small benchmark dataset.

## 📈 Interactive Dashboard

The Streamlit dashboard includes six sections:

1. Executive Overview
2. Data Explorer
3. Predict Studio
4. Batch Prediction
5. Model Intelligence
6. About Project

### Dashboard Screenshots

Add your dashboard screenshots to a `screenshots` folder and display them here.

<!-- Add screenshots here after capturing them -->

## 🗂️ Project Structure

```text
Iris-Intelligence-Pro/
│
├── app.py
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

- Total observations: 150
- Input features: 4
- Target classes: 3

Input features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Measurements are recorded in centimetres.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Jupyter Notebook
- Pytest

## 🔍 Methodology

1. Load and validate the dataset.
2. Perform exploratory data analysis.
3. Split the dataset into training and testing sets using stratified sampling.
4. Tune five classification models using GridSearchCV.
5. Apply stratified 5-fold cross-validation.
6. Select the best model using cross-validation accuracy.
7. Evaluate the selected model on the held-out test set.
8. Save the trained model and evaluation outputs.
9. Use the trained model for interactive predictions.

Feature scaling is performed inside pipelines to prevent data leakage during cross-validation.

## 🎯 Learning Outcomes

Through this project, I worked on:

- End-to-end Machine Learning workflow
- Classification algorithms
- Model comparison and hyperparameter tuning
- Cross-validation and model evaluation
- Data visualization
- Streamlit dashboard development
- Python project organization
- Automated testing

## ⚠️ Limitations

The Iris dataset is a small benchmark dataset. Its results do not guarantee similar performance on new datasets or real-world measurement systems.

The test set contains only 30 observations. A single incorrect prediction changes test accuracy by approximately 3.33 percentage points.

## 👨‍💻 Author

**Srishti Pandey**

MCA Data Science Student

GitHub: [krishankantpandey359-droid](https://github.com/krishankantpandey359-droid)

---

⭐ If you find this project useful, feel free to explore the repository.
