
# Spam Filtering Email System Using Machine Learning

This project develops an email spam filtering system using Linear, Logistic Regression and Perceptron classifiers. It processes email data, trains models, and evaluates performance with metrics like accuracy, F1-score, and ROC-AUC curves. The notebook also features Exploratory Data Analysis (EDA) with the Sweetviz library for automated insights.


## Authors

- Mubanga Nsofu [@RproDigest](https://github.com/RProDigest/) 


## Badges


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Key Features

**1. Dynamic File Loading:**

Allows users to select a dataset file dynamically through a Tkinter file dialog.

**2. Exploratory Data Analysis (EDA):**

Generates a comprehensive HTML report using sweetviz.

**3. Preprocessing:**

Converts text data into numerical features using CountVectorizer.
Splits data into training and testing sets.

**4. Machine Learning Models:**
Implements Linear Regression and Logistic Regression-based Perceptrons.

**5. Unit Testing:**
The notebook includes unit testing to validate model components and ensure that minimum accuracy thresholds are achieved. Three-unit tests are performed for the three models.

**6. Performance Evaluation:**
Classification metrics: Accuracy, F1-score, and classification reports.
Visualizations: Bar charts for accuracy and F1-scores, and ROC curves with AUC.


## Files In Repository

### Milestone_One_Assignment_Mubanga_Nsofu_149050_BAN6440.ipynb:

The Jupyter Notebook contains the project's implementation. The steps include:
- Installation &  loading of python libraries
- Selection of Downloaded file 
- Exploratory Data Analysis
- Data preprocessing
- Unit Testing
- Model Definition & Evaluation
- Model Performance visualisation

### email.csv:

The dataset is from kaggle and can be found at this link:

https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data 

### EDA_Report.html:
 An automatically generated HTML report by sweetviz.

### Summary PPtx
Powerpoint summary of the project called "*Milestone Assignment BAN6440 M_Nsofu_149050.pptx*"


### README.md: 

This documentation file explaining the project, its objectives, and implementation.
## Project Workflow

### 1. Load Data:
The user selects a CSV file containing the email dataset with columns text (email content) and spam (0 for non-spam, 1 for spam).

### 2. EDA:
Generates a detailed HTML report (EDA_Report.html) for understanding data distributions, missing values, and correlations.

### 3. Preprocessing:
Prepares the data by splitting it into training and testing sets and vectorizing the text content.

### 4 Train Models:
Trains Three classifiers:
- Linear Regression
- Logistic Regression
- Perceptron

### 5 Evaluate Models:

- Outputs classification reports with accuracy, precision, recall, and F1-scores.
- Generates ROC curves with AUC values.

## 6 Unit Testing:
Confirms that models meet predefined performance thresholds (90% for Linear Regression, 95% for Logistic Regression and 85% for the Perceptron).

## 7. Visualize Results:
Plots accuracy and F1-scores as bar charts and includes ROC curves.
## How to Run the Notebook

### 1.0 Install dependencies:

 Please ensure all required libraries are installed by running the line below.

``` python
pip install pandas numpy scikit-learn matplotlib sweetviz unittest tkinter 
```

or in the notebook ensure to run the line below

``` python
!pip install pandas numpy scikit-learn matplotlib sweetviz unittest tkinter
```

### 2.0 Launch Jupyter Notebook in your IDE of choice

### 3.0 Execute the Cells: 

Run the notebook cells sequentially to:

- Load the data set.
- Perform exploratory data analysis.
- Preprocess the data.
- Train the linear and logistic regression models.
- Evaluate and visualize results.
- Perform three unit tests


### 4.0 View Results:

Performance metrics will be displayed within the notebook.



## Expected Project Results

**1. Linear Regression**

**Accuracy: 93.46%**

- 93.46% of the predictions made by the Linear Regression Perceptron are correct.

**Precision:**

- Class 0 (Not Spam): 0.95 (95% of predicted non-spam emails were actually non-spam).
- Class 1 (Spam): 0.89 (89% of predicted spam emails were actually spam).

**Recall:**

- Class 0 (Not Spam): 0.97 (97% of actual non-spam emails were correctly identified as non-spam).
- Class 1 (Spam): 0.84 (84% of actual spam emails were correctly identified as spam).

**F1-Score:**

- Class 0 (Not Spam): 0.96 (A balance between precision and recall for non-spam).
- Class 1 (Spam): 0.87 (A balance between precision and recall for spam).

**Macro Avg:** 0.91 (Averaged equally across classes).
**Weighted Avg:** 0.93 (Weighted by the number of samples in each class).


**2. Logistic Regression**

**Accuracy: 99.13%**

- 99.13% of the predictions made by the Logistic Regression Perceptron are correct.

**Precision:**

- Class 0 (Not Spam): 0.99 (99% of predicted non-spam emails were actually non-spam).
- Class 1 (Spam): 0.99 (99% of predicted spam emails were actually spam).

**Recall:**

- Class 0 (Not Spam): 1.00 (100% of actual non-spam emails were correctly identified as non-spam).
- Class 1 (Spam): 0.97 (97% of actual spam emails were correctly identified as spam).

**F1-Score:**

- Class 0 (Not Spam): 0.99.
- Class 1 (Spam): 0.98.

**Macro Avg:** 0.99 (Averaged equally across classes).

**Weighted Avg:** 0.99 (Weighted by the number of samples in each class).

**3. Perceptron**

Perceptron Classifier Accuracy: 0.9851657940663177
Perceptron Classifier Report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99       856
           1       0.98      0.96      0.97       290

    accuracy                           0.99      1146
   macro avg       0.98      0.98      0.98      1146
weighted avg       0.99      0.99      0.99      1146


## Future Enhancements

1. Use real-world datasets for more practical insights.
2. Evaluate the model's robustness with additional test datasets.
## References

Jackksoncsie. (2023). Spam email Dataset. https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data

GeeksforGeeks. (2023). *Logistic Regression using Python*. https://www.geeksforgeeks.org/ml-logistic-regression-using-python/

DataCamp. (2024). *Python Logistic Regression Tutorial with Sklearn & Scikit*. https://www.datacamp.com/tutorial/understanding-logistic-regression-python