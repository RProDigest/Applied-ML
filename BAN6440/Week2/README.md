
# Optuna-Tuned XGBoost Model for Inventory Management

This project uses synthetic data to demonstrate how to apply a tuned XGBoost Model for effective Inventory Management in the fast-moving consumer goods sector (FMGC). XGboost is tuned using Optuna.
The project is implemented in a Jupyter Notebook, demonstrating feature engineering, hyperparameter tuning, and model evaluation using synthetic data.


## Authors

- Mubanga Nsofu [@RproDigest](https://github.com/RProDigest/) 


## Badges


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Project Overview

### Objective:

- Predict the number of Days_to_Stock_out for FMCG products based on various inventory and sales features. This will help Amway to make smarter inventory stock management decisons and minimize stock holding costs

- Optimise model performance using Bayesian hyperparameter tuning with Optuna.

### Dataset:

Synthetic data generated programmatically with 300 records, including inventory levels, sales volume, demand variance, and other relevant features.

### Machine Learning Approach:

#### Solutions: 
- Model: XGBoost (Extreme Gradient Boosting)
- Optimization Tool: Optuna for hyperparameter tuning
- Evaluation Metrics: Mean Squared Error (MSE) and R-squared (R²)

## Files In Repository

### MubangaNsofu_Assignment_code.ipynb:

The Jupyter Notebook contains the project's implementation. The steps include:
- Data generation and preprocessing
- Feature engineering
- Hyperparameter tuning with Optuna
- Model evaluation
- Feature importance visualisation
- Model saving

### synthetic_inventory_data.csv:

The synthetic dataset used for training and testing the model, generated within the notebook.

### Trained and Tuned XGBoost Model 
- optuna_tuned_xgboost_model.pkl

This is the final trained XGBoost model saved for deployment or further use.

### README.md: 

This documentation file explaining the project, its objectives, and implementation.
## Project Workflow

### Step 1: Data Preparation

- **Synthetic Data Generation:**
Using numpy functions, a synthetic dataset is created. The features include:

    - Monthly_Sales_Volume 
    - Inventory_Levels 
    - Reorder_Point
    - Demand_Variance 
    - Days_to_Stock_out
    - Lead_Time
    - Price_Per_Unit
    - Supplier_Reliability_Score
    - Product_ID
    - Product_Category
    
The Target variable (Days_to_Stock_out) is calculated with added noise to simulate real-world scenarios.
- **Feature Engineering:**
Derived features such as Inventory_to_Sales_Ratio and Effective_Demand have been added to enhance the model's predictive power.
- **Data Preprocessing:**
Categorical variables are encoded using one-hot encoding.

### Step 2: Model Training
Before training the model, the synthetic data is split into training (70%) and testing (30%) sets.


### Step 3: Hyperparameter Tuning
Use Optuna’s Bayesian Optimization to fine-tune the following parameters:
- Number of estimators (n_estimators)
- Learning rate (learning_rate)
- Maximum tree depth (max_depth)
- Subsample ratio (subsample)
- Column sampling ratio (colsample_bytree)
- Run 50 trials to minimize the Mean Squared Error (MSE).

### Step 4: Model Evaluation
Evaluate the tuned model on the test set using:
- MSE: Measures prediction error.
- R²: Explains the variance captured by the model.

### Step 5: Feature Importance Visualization
Visualize the relative importance of features using a horizontal bar chart generated using matplotlib.

### Step 6: Model Saving
Save the trained model as final_optuna_tuned_xgboost_model.pkl for future use.

## How to Run the Notebook

### 1.0 Install dependencies:

 Please ensure all required libraries are installed by running the line below.

``` python
pip install pandas numpy optuna scikit-learn matplotlib xgboost joblib notebook
```

or in a notebook

``` python
!pip install pandas numpy optuna scikit-learn matplotlib xgboost joblib notebook
```

### 2.0 Open the Notebook: Launch Jupyter Notebook in your IDE of choice

### 3.0 Execute the Cells: 

Run the notebook cells sequentially to:

- Generate synthetic data.
- Perform feature engineering.
- Train and optimize the XGBoost model.
- Evaluate and visualize results.


### 4.0 View Results:

Performance metrics and feature importance plots will be displayed within the notebook.

### 5.0 Access the Model:

The trained model is saved as final_optuna_tuned_xgboost_model.pkl.


## Project Results


- Mean Squared Error (MSE): 0.15998

The average squared difference between predicted and actual values is minimal (MSE of 0.15998), showing that the model predicts Days_to_Stock_out with high accuracy.

- R-squared (R²): 0.8442

The model explains 84.42% (a significant proportion) of the variance in the target variable (Days_to_Stock_out). XGBoost effectively captures the complex non-linear aspects of the dataset. 

- Summary: The model effectively predicts Days_to_Stock_out, with over 84% of variance explained.
## Future Enhancements

1. Use real-world datasets for more practical insights.
2. Experiment with additional machine learning models (e.g., LightGBM, CatBoost).
3. Increase the number of Optuna trials for finer hyperparameter tuning.
4. Evaluate the model's robustness with additional test datasets.
## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 13-17-August-2016*, 785–794. https://doi.org/10.1145/2939672.2939785

Nvidia. (2024). XGBoost – *What Is It and Why Does It Matter?* https://www.nvidia.com/en-us/glossary/xgboost/

Tin Kam Ho. (1995). Random decision forests. *Proceedings of 3rd International Conference on Document Analysis and Recognition*, 1, 278–282. https://doi.org/10.1109/ICDAR.1995.598994

Rocca, J. (2019). *Ensemble methods: bagging, boosting and stacking* – Towards Data Science. https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205
