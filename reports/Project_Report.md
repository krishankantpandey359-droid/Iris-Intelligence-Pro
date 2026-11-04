# Iris Flower Classification: Technical Project Report

## Objective
Predict the species of an Iris flower from sepal length, sepal width, petal length and petal width.

## Dataset
The scikit-learn Iris benchmark contains 150 observations, four continuous input variables and three balanced target classes. Each class has 50 observations.

## Data validation
The project checks schema completeness, numeric feature types, missing values, target availability and duplicate-row count. Duplicate measurements are reported rather than automatically deleted. Identical feature values alone do not prove a source-record duplication.

## Exploratory analysis
The analysis covers descriptive statistics, class distribution, feature histograms, box plots, pairwise feature relationships and a correlation matrix. Petal measurements often provide strong separation among the classes in this benchmark.

## Modelling
Five algorithms are tuned using GridSearchCV and stratified five-fold cross-validation. Scaling is contained inside pipelines for Logistic Regression, KNN and SVM. The split is stratified and uses a fixed random seed.

## Evaluation
Model selection uses cross-validation accuracy on the training partition. The held-out test partition is used for accuracy, macro precision, macro recall, macro F1, classification report and confusion matrix.

## Limitations
This small, curated benchmark does not represent all real-world flower populations or measurement conditions. A random split estimates performance under a similar sampling process. External validation is required before operational deployment.

## Conclusion
The project demonstrates a reproducible classification workflow, model tuning, evaluation, artifact generation, automated tests and an interactive prediction dashboard. Numeric claims must be taken from the generated evaluation files, not assumed in advance.
