# OIBSIP Data Science Internship
## Task 1: Iris Flower Classification

A complete end-to-end machine-learning project for multiclass Iris species classification.

### What is included
- Reproducible dataset loading from scikit-learn.
- Data validation and exploratory data analysis.
- Five classification algorithms.
- Hyperparameter tuning with GridSearchCV.
- Stratified five-fold cross-validation.
- Held-out test evaluation.
- Accuracy, macro precision, macro recall, macro F1, classification report and confusion matrix.
- Saved model pipeline and evaluation outputs.
- Interactive Streamlit dashboard.
- Automated data validation tests.

### Dataset
The canonical Iris dataset contains 150 observations, four numeric features measured in centimetres, and three classes: setosa, versicolor and virginica. The dataset is bundled with scikit-learn. No manual dataset download or file placement is required.

### Run the project
Use Python 3.10 or newer.

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py
pytest -q
streamlit run app.py
```

The commands must be run from the extracted project folder. The first training run creates the saved model and evaluation files. The Streamlit app also trains the model automatically if the saved model is absent.

### Project methodology
1. Load and validate the data.
2. Explore distributions, class balance, feature relationships and correlations.
3. Split data into stratified training and test sets (80:20).
4. Tune five models using stratified five-fold cross-validation on the training set.
5. Select the best model using cross-validation accuracy.
6. Evaluate once on the held-out test set.
7. Save model and evaluation artifacts.

### Models
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest
- Gradient Boosting

Scaling is performed inside pipelines for scale-sensitive models. This prevents preprocessing leakage across cross-validation folds.

### Important interpretation
The Iris dataset is a small, balanced benchmark. A high score does not guarantee performance on new measurement devices, regions or populations. The test set contains only 30 observations, so one incorrect prediction changes accuracy by about 3.33 percentage points. Probability scores are not guaranteed to be calibrated.

### Folder structure
```text
OIBSIP_Task1_Iris_Classification/
├── app.py
├── README.md
├── requirements.txt
├── data/iris.csv
├── notebooks/Task1_Iris_Flower_Classification.ipynb
├── src/data_utils.py
├── src/train.py
├── tests/test_project.py
├── models/
├── outputs/
└── reports/Project_Report.md
```

### Before recording your internship demo
- Run the training command and tests.
- Launch the Streamlit app.
- Show the dataset overview, EDA, model comparison and a live prediction.
- Quote only the metrics generated in your own run.
