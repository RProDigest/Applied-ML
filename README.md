
# Otomoto Marketing Segmentation Model Optimization
## Overview
This project focuses on optimizing an Artificial Neural Network (ANN) to improve marketing segmentation for Otomoto. By leveraging customer data, the optimized model aims to enhance marketing campaign effectiveness through better customer segmentation.

## Authors

- Mubanga Nsofu [@RproDigest](https://github.com/RProDigest/) 


## Badges


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Dataset

- Provided in the assignment question and included in the github repository. The dataset is called teleconnect.csv

- Preprocessing:
    -  Handling missing values.
    -  Encoding categorical features using LabelEncoder.
    -  Splitting the data into training and testing sets.
## Key Features

- Baseline ANN Model: A simple ANN trained using SMOTE to address class imbalance.
- Optimized ANN Model:
    - Implements undersampling for balancing classes.
    - Scales features using StandardScaler.
    - Includes Dropout regularization to prevent overfitting.
    - Utilizes bias initialization to stabilize training.
    - Incorporates EarlyStopping to improve training efficiency.
- Output Layer: Single neuron with sigmoid activation for binary classification.
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy



## Implementation

**1. Baseline Model:**

- Architecture: Three hidden layers.

- Optimization: SMOTE for class imbalance.

- Metrics: Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

**2. Optimized Model:**

- Architecture: Three hidden layers with Dropout.

- Optimization: Undersampling, feature scaling, bias initialization, and EarlyStopping.

- Metrics: Same as the baseline.
## Evaluation Approach

Both models are evaluated using:

- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC.

- Visualizations:

    - Training and validation loss.

    - ROC curves.

    - Precision-recall curves.

    - Confusion matrices.
## Usage

**1. Requirements**

- Python 3.12 or later
 Required libraries:
- pandas
- numpy
- matplotlib
- tensorflow
- scikit-learn
- seaborn
- imbalanced-learn


**2. Install dependencies**

Install the required Python libraries using pip:

```bash
pip install <library name1> <library name2> 
```
Example

```bash
pip install tensorflow scikit-learn pandas numpy imbalanced-learn matplotlib seaborn
```

**3. Run the application**

Run the Jupyter notebook sequentially



**4. Outputs**
-  View performance metrics and visualizations such as Accuracy, Recall, AUC-ROC,  Confusion Matrix 

- The optimised model should outperform the baseline model over several runs


## References

Google. (2024). Classification on imbalanced data. TensorFlow Core. https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

Tamanna. (2023). Handling Imbalanced Datasets in Python: Methods and Procedures. Medium. https://medium.com/@tam.tamanna18/handling-imbalanced-datasets-in-python-methods-and-procedures-7376f99794de</p

Tang, T. (2023). Class Imbalance Strategies — A Visual Guide with Code. Towards Data Science. https://towardsdatascience.com/class-imbalance-strategies-a-visual-guide-with-code-8bc8fae71e1a

Jin, H., Song, Q., & Hu, X. (2019). Auto-Keras: An Efficient Neural Architecture Search System. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 1946–1956. https://doi.org/10.1145/3292500.3330648

GunKurnia. (2024). Selecting Independent Features: Balancing Feature Selection, SHAP Values, and Domain Knowledge for Better Model Accuracy. Medium. https://medium.com/@gunkurnia/selecting-independent-features-balancing-feature-selection-shap-values-and-domain-knowledge-for-d3231a332a9c

