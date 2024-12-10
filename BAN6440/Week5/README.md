
# Breast Cancer Diagnosis Model
Using the Breast Cancer Wisconsin dataset, this project builds and evaluates a deep-learning model to classify breast cancer cases as benign or malignant. The model is designed to assist in early and accurate cancer diagnosis, potentially aiding in large-scale screening and telemedicine applications.

## Authors

- Mubanga Nsofu [@RproDigest](https://github.com/RProDigest/) 


## Badges


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Dataset

- Name: Breast Cancer Wisconsin Dataset
- Source: Kaggle or UCI Machine Learning Reposity
- Features: 30 numeric attributes related to cell nuclei characteristics
- Target Variable: - 0: Benign,  1: Malignant

## Model Architecture

- Input Layer: 30 features (after preprocessing and scaling).
- Hidden Layers:
    - Dense layers with LeakyReLU activation.
    - Dropout for regularization.
- Output Layer: Single neuron with sigmoid activation for binary classification.
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
## Usage

**1. Requirements**

- Python 3.12 or later
 Required libraries:
- pandas
- numpy
- matplotlib
- tensorflow
- scikit-learn
- sweetViz



**2. Install dependencies**

Install the required Python libraries using pip:

```bash
pip install <library name1> <library name2> 
```

**3. Run the application**

Run the Jupyter notebook sequentially



**4. Outputs**
- Accuracy and classification reports for each fold in the notebook cell
- Training and validation loss and accuracy plots
- Sweetviz EDA report in EDA_Report.html

## Evaluation Approach

1. Data Preprocessing:

    - Removed irrelevant columns (id and Unnamed: 32).
    - Mapped target values to binary labels.
    - Standardized feature values using StandardScaler.
2. K-Fold Cross-Validation:

    - Stratified splitting to maintain class balance across folds.
    - 5 folds used for training and evaluation.
    - Metrics computed:
        - Accuracy
        - Precision, Recall, F1-Score (per fold)
3. Performance Metrics:

    - Average accuracy across folds.
    - Training and validation loss/accuracy plots.
## Expected Project Results

1. Accuracy:

- Mean Accuracy: ~99%

- Classification Metrics:

    -High precision, recall, and F1-scores for both benign and malignant classes.

- Training vs. Validation:

    - Validation loss consistently decreased, aligning closely with training loss, indicating good generalization.
## References

Tran, K. A., Kondrashova, O., Bradley, A., Williams, E. D., Pearson, J. v., & Waddell, N. (2021). Deep learning in cancer diagnosis, prognosis and treatment selection. Genome Medicine, 13(1), 152. https://doi.org/10.1186/s13073-021-00968-x


google. (n.d.). Keras: The high-level API for TensorFlow | TensorFlow Core. Retrieved December 9, 2024, from https://www.tensorflow.org/guide/keras


Kaul, S. (2024). Building Deep Learning Models with Keras: A Step-by-Step Guide with Code Examples. Medium. https://medium.com/@sumit.kaul.87/building-deep-learning-models-with-keras-a-step-by-step-guide-with-code-examples-68aee4152625

