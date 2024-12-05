
# Phrase Clustering Application Using Machine Learning

This project demonstrates the implementation of K-Means clustering on a phrase dataset to uncover distinct patterns in the text. It includes data preprocessing, feature extraction, clustering, visualization, and unit testing, orchestrated through a centralized main.py script.


## Authors

- Mubanga Nsofu [@RproDigest](https://github.com/RProDigest/) 


## Badges


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Dataset

The dataset used for this project is the Phrase Clustering Dataset (PCD). It is downloaded from AWS S3 using the AWS CLI with no-sign-request.
## Key Features

**1. Data Loading:**
 
 Downloads the dataset from AWS S3 using main.py.

**2. Feature Extraction:** 

Extracts word and character counts from textual phrases.

**3. K-Means Clustering:** 

Identifies distinct phrase clusters using clustering.py.

**4.Visualization:** 

- Produce scatter plots
- PCA-based visualization

**Unit Testing:** 

Conducts automated tests of core functions via test_clustering.py, with results saved in the unit_test_results folder.
## Usage

**1. Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>

```

**2. Requirements**

- Python 3.12 or later
 Required libraries (in requirements.txt):
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn





**3. Install dependencies**

Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

**4. Run the application**

Use the main.py script to orchestrate the entire workflow, including data download, clustering, visualization, and testing

```bash

python main.py
```



## Expected Project Results

- Clustering Visualizations: Scatter plots and PCA-based plots
- Unit Test Results: Logs and reports saved in the unit_test_results folder.
## References

Phrase Clustering Dataset. (n.d.). Phrase Clustering Dataset (PCD). Retrieved December 4, 2024, from https://registry.opendata.aws/pcd.