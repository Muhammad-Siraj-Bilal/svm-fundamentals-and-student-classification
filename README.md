<div align="center">

# Support Vector Machines: Fundamentals & Student Grade Classification

### From visual decision boundaries to a real-world classification application

**Python · Scikit-learn · Machine Learning · Jupyter Notebook**

</div>

---

## Overview

This repository provides a practical exploration of **Support Vector Machines**, progressing from simple two-dimensional decision boundaries to non-linear kernel transformations and a real-world student-grade classification problem.

The notebooks are organised as a learning journey:

1. Visualise a linear decision boundary
2. Understand positive and negative decision regions
3. Explore non-linear transformations and SVM kernels
4. Apply an RBF-based SVM to student-performance data

The project focuses on both the mathematical intuition behind SVMs and their practical implementation using Python and Scikit-learn.

---

## What Is a Support Vector Machine?

A Support Vector Machine is a supervised machine-learning algorithm commonly used for classification.

Its objective is to find a decision boundary that separates classes while maximising the distance between the boundary and the nearest observations.

The closest observations are known as **support vectors**.

For a linear classifier, the decision boundary can be represented as:

[
w^T x + b = 0
]

Where:

* (w) controls the orientation of the boundary
* (x) represents the input features
* (b) controls the position of the boundary

SVMs can also classify non-linear data by using kernel functions that map observations into a higher-dimensional feature space.

---

## Repository Structure

```text
svm-fundamentals-and-student-classification/
│
├── 01_svm_decision_line.ipynb
├── 02_svm_decision_regions.ipynb
├── 03_svm_kernel_transformations.ipynb
├── 04_student_grade_classification.ipynb
│
├── data/
│   ├── non_linear.csv
│   └── student-mat.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notebook Guide

### 1. Linear SVM Decision Line

```text
01_svm_decision_line.ipynb
```

This notebook introduces the fundamental idea of separating two classes using a straight decision line.

It demonstrates:

* Plotting two-dimensional observations
* Representing positive and negative classes
* Drawing a linear decision boundary
* Understanding how the line separates classes
* Visualising the geometry of binary classification

This notebook establishes the basic intuition required to understand linear Support Vector Machines.

---

### 2. SVM Decision Regions

```text
02_svm_decision_regions.ipynb
```

This notebook extends the decision-line example by visualising the complete classification space.

It demonstrates:

* Positive and negative decision regions
* Classification on both sides of a boundary
* Mesh-grid generation
* Region-based prediction visualisation
* The relationship between a decision function and its output classes

Instead of showing only the separating line, the notebook highlights which areas of the feature space belong to each class.

---

### 3. SVM Kernel Transformations

```text
03_svm_kernel_transformations.ipynb
```

Some datasets cannot be separated using a straight line. This notebook explores how mathematical transformations can make non-linear data easier to classify.

It covers:

* Polynomial transformation
* Radial Basis Function transformation
* Sigmoid transformation
* Non-linear feature mapping
* Higher-dimensional class separation
* Visual comparison of transformed feature spaces

### Polynomial Kernel

The polynomial kernel allows the model to learn curved relationships by introducing polynomial combinations of the original features.

[
K(x_i,x_j)=(\gamma x_i^T x_j+r)^d
]

### Radial Basis Function Kernel

The RBF kernel measures similarity based on the distance between observations.

[
K(x_i,x_j)=e^{-\gamma \lVert x_i-x_j \rVert^2}
]

It is particularly useful when the relationship between classes is complex and non-linear.

### Sigmoid Kernel

The sigmoid kernel behaves similarly to an activation function used in neural networks.

[
K(x_i,x_j)=\tanh(\gamma x_i^T x_j+r)
]

---

### 4. Student Grade Classification

```text
04_student_grade_classification.ipynb
```

This notebook applies Support Vector Machines to a student-performance dataset.

The objective is to classify students into two final-grade categories:

| Class          | Condition  |
| -------------- | ---------- |
| High grade     | `G3 > 10`  |
| Moderate grade | `G3 <= 10` |

The notebook demonstrates a complete machine-learning workflow.

#### Workflow

1. Load the student-performance dataset
2. Inspect the data and target distribution
3. Create the binary `High_G3` target
4. Remove the original `G3` value from the predictors
5. Separate features and labels
6. One-hot encode categorical variables
7. Standardise numerical features
8. Create stratified training and testing sets
9. Train an RBF Support Vector Classifier
10. Train a class-balanced SVM
11. Generate predictions
12. Compare model performance
13. Display confusion matrices
14. Analyse precision, recall and F1-score

---

## Standard and Balanced SVM Models

The student-classification notebook compares two approaches.

### Standard SVM

The standard model treats the available training observations normally.

```python
SVC(kernel="rbf")
```

### Class-Balanced SVM

The balanced model gives additional importance to the less frequent class.

```python
SVC(
    kernel="rbf",
    class_weight="balanced"
)
```

This can help the model recognise minority-class observations instead of favouring the larger class.

---

## Machine-Learning Pipeline

The student-performance analysis includes several important preprocessing stages.

### Categorical Encoding

Categorical fields are transformed through one-hot encoding so that they can be processed by the model.

Example variables may include:

* School
* Gender
* Address type
* Family size
* Parental status
* School support
* Family support
* Internet access
* Extra-curricular activities

### Feature Scaling

SVMs are sensitive to differences in feature magnitude.

Standardisation transforms numerical values so that they have approximately:

* Mean of 0
* Standard deviation of 1

This prevents variables with larger numerical ranges from dominating the classifier.

### Stratified Data Split

A stratified train-test split preserves the proportion of each target class in both datasets.

This produces a more representative evaluation.

---

## Model Evaluation

The student-grade classifiers are evaluated using:

### Confusion Matrix

The confusion matrix displays:

* True positives
* True negatives
* False positives
* False negatives

### Accuracy

Accuracy measures the proportion of all predictions that were correct.

[
Accuracy =
\frac{Correct\ Predictions}{Total\ Predictions}
]

### Precision

Precision measures how many predicted positive cases were actually positive.

[
Precision =
\frac{TP}{TP+FP}
]

### Recall

Recall measures how many actual positive cases were correctly identified.

[
Recall =
\frac{TP}{TP+FN}
]

### F1-Score

The F1-score balances precision and recall.

[
F1 =
2 \times
\frac{Precision \times Recall}
{Precision+Recall}
]

These metrics provide a more complete view of classifier performance than accuracy alone.

---

## Key Concepts Demonstrated

* Binary classification
* Linear decision boundaries
* Decision regions
* Maximum-margin classification
* Support vectors
* Non-linear transformations
* Kernel methods
* Polynomial kernels
* RBF kernels
* Sigmoid kernels
* Categorical encoding
* Numerical standardisation
* Stratified train-test splitting
* Class balancing
* Confusion matrices
* Classification reports
* Precision, recall and F1-score

---

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Muhammad-Siraj-Bilal/svm-fundamentals-and-student-classification.git
cd svm-fundamentals-and-student-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file containing:

```text
jupyter
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Dataset Setup

Place the required datasets inside a `data` directory:

```text
data/
├── non_linear.csv
└── student-mat.csv
```

Use repository-relative paths inside the notebooks:

```python
import pandas as pd

non_linear_data = pd.read_csv("data/non_linear.csv")
student_data = pd.read_csv("data/student-mat.csv")
```

When the notebooks are placed inside a separate `notebooks` folder, use:

```python
student_data = pd.read_csv("../data/student-mat.csv")
```

---

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open and run the notebooks in this order:

```text
01_svm_decision_line.ipynb
02_svm_decision_regions.ipynb
03_svm_kernel_transformations.ipynb
04_student_grade_classification.ipynb
```

Run each cell using:

```text
Shift + Enter
```

The notebooks can also be opened through:

* JupyterLab
* Visual Studio Code
* Google Colab

---

## Learning Outcomes

Through this repository, users can learn how to:

* Understand the intuition behind Support Vector Machines
* Visualise binary decision boundaries
* Interpret positive and negative classification regions
* Understand why linear separation may fail
* Transform non-linear data into separable representations
* Compare polynomial, RBF and sigmoid transformations
* Build a complete SVM classification workflow
* Prepare mixed categorical and numerical data
* Handle imbalanced classes
* Evaluate classifiers using multiple performance metrics
* Apply SVMs to a practical education-related dataset

---

## Potential Improvements

Future development could include:

* Hyperparameter tuning with GridSearchCV
* Comparison of linear, polynomial, RBF and sigmoid SVM kernels
* Cross-validation
* ROC and AUC analysis
* Precision-recall curves
* Learning curves
* Feature-selection experiments
* Principal Component Analysis
* Interactive decision-boundary visualisations
* Streamlit-based student-classification interface
* Model persistence using Joblib
* Explainability using permutation importance or SHAP

---

## Educational Purpose

This repository was developed for educational and portfolio purposes to demonstrate the progression from basic SVM intuition to practical machine-learning implementation.

The student-grade predictions are intended as a technical classification exercise and should not be used as the sole basis for real academic decisions.

---

## Author

**Muhammad Siraj Bilal**

---

## Licence

This repository is intended for educational and research purposes.

Review the original dataset licence before redistributing the included data.

---

<div align="center">

### Support Vector Machines, visualised and applied.

</div>
