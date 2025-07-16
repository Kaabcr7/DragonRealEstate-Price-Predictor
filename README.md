# 🏡 Dragon Real Estate - Price Predictor

## 📌 Project Overview

This project implements a machine learning model to predict housing prices, leveraging a dataset that includes various socio-economic factors and property attributes. The core of this predictor is a **Random Forest Regressor**.

The primary objective is to build a reliable predictive model that can estimate property sale prices—useful for real estate valuation, market analysis, or informing investment strategies.

---

## 🚀 Features

* **Data Loading & Initial Exploration:** Uses `head`, `info`, `describe`, `value_counts` to explore the dataset.
* **Data Visualization:** Scatter plots and scatter matrix for relationship visualization.
* **Attribute Combination:** Creates new features (e.g., `TAXRM`) by combining existing ones.
* **Train-Test Splitting:** StratifiedShuffleSplit ensures proper distribution (especially for skewed `CHAS`).
* **Missing Value Imputation:** Uses `SimpleImputer` with a median strategy.
* **Feature Scaling:** Applies `StandardScaler`.
* **Pipeline Integration:** Combines preprocessing steps using `Pipeline`.
* **Random Forest Regressor:** Chosen for its high performance in regression tasks.

---

## 📊 Model Evaluation

* RMSE on training data
* 10-Fold Cross-Validation using `cross_val_score` for robust error estimation

---

## 💾 Model Persistence

Saves the model and preprocessing pipeline using `joblib`.

---

## 🔍 Prediction Module

Demonstrates how to load the saved model and make predictions on new, unseen data.

---

## 🛠️ Technologies Used

* **Python 3.x**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Visualization
* **Scikit-learn** – ML algorithms, preprocessing, pipelines
* **Joblib** – Model persistence

---

## 📂 Dataset

This project uses a `data.csv` file in the root directory, similar to the **Boston Housing Dataset**:

| Column  | Description                                                         |
| ------- | ------------------------------------------------------------------- |
| CRIM    | Per capita crime rate by town                                       |
| ZN      | Proportion of residential land zoned for lots over 25,000 sq.ft.    |
| INDUS   | Proportion of non-retail business acres per town                    |
| CHAS    | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX     | Nitric oxides concentration (parts per 10 million)                  |
| RM      | Average number of rooms per dwelling                                |
| AGE     | Proportion of owner-occupied units built prior to 1940              |
| DIS     | Weighted distances to employment centers                            |
| RAD     | Index of accessibility to radial highways                           |
| TAX     | Property-tax rate per \$10,000                                      |
| PTRATIO | Pupil-teacher ratio by town                                         |
| B       | 1000(Bk−0.63)^2, where Bk is the proportion of Black residents      |
| LSTAT   | % lower status of the population                                    |
| MEDV    | Median value of homes in \$1000s (Target)                           |

⚠️ Ensure your `data.csv` matches this structure.

---

## 🧪 Installation and Setup

1. Clone the Repository:
   `git clone https://github.com/your-username/DragonRealEstate-Price-Predictor.git`
   `cd DragonRealEstate-Price-Predictor`

2. Add Dataset:
   Place your `data.csv` file in the root directory of the project.

3. Create a `requirements.txt` file with the following content:
   pandas
   numpy
   matplotlib
   scikit-learn
   joblib

4. Set Up a Virtual Environment (recommended):
   `python -m venv venv`

5. Activate the Virtual Environment:
   On Windows: `.\venv\Scripts\activate`
   On macOS/Linux: `source venv/bin/activate`

6. Install the dependencies:
   `pip install -r requirements.txt`

7. (Optional) Install Jupyter Notebook:
   `pip install jupyter`

---

## ▶️ How to Run the Project

1. Launch Jupyter Notebook:
   `jupyter notebook`

2. Open the notebook file: `DragonRealEstate.ipynb`

3. Execute each cell from top to bottom to:

   * Load and explore the dataset
   * Preprocess and split the data
   * Train the Random Forest model
   * Evaluate the model using RMSE and cross-validation
   * Save the trained model and pipeline
   * Make predictions on new, unseen data

---

## 📈 Results

After running the notebook, you will see:

* Initial Data Overview: Summary statistics, feature types, and any missing values
* Correlation Analysis: Insights into how each feature correlates with the target variable (`MEDV`)
* Cross-Validation Scores: Root Mean Squared Error (RMSE) from 10-fold cross-validation
* Final Model Performance: RMSE on the test set to evaluate generalization

### Example Metrics

* Cross-validation RMSE (mean): \~2.91
* Final test RMSE: \~2.91

These values suggest that the model has good predictive accuracy on this dataset.

---

## 💾 Saving and Loading the Model

To save the pipeline and model:

```python
from joblib import dump
dump(my_pipeline, 'Dragon_pipeline.joblib')
dump(model, 'Dragon_model.joblib')
```

To load and make predictions:

```python
from joblib import load
import pandas as pd

my_pipeline = load('Dragon_pipeline.joblib')
model = load('Dragon_model.joblib')

new_data = pd.DataFrame({
    'CRIM': [0.00632], 'ZN': [18.0], 'INDUS': [2.31], 'CHAS': [0],
    'NOX': [0.538], 'RM': [6.575], 'AGE': [65.2], 'DIS': [4.09],
    'RAD': [1], 'TAX': [296], 'PTRATIO': [15.3], 'B': [396.90], 'LSTAT': [4.98]
})

processed_data = my_pipeline.transform(new_data)
predicted_price = model.predict(processed_data)
print(f"Predicted price: ${predicted_price[0]:.2f} (in $1000s)")
```

---

## 🌱 Future Enhancements

* Hyperparameter Tuning using `GridSearchCV` or `RandomizedSearchCV`
* More Feature Engineering and transformation pipelines
* Outlier Detection and handling strategies
* Model Comparison with other regressors such as `LinearRegression`, `XGBoost`, `SVR`, etc.
* Residual Analysis and error distribution visualization
* Deployment with Flask, FastAPI, or Streamlit
* Interactive Data Visualizations with Plotly or Bokeh

---


Feel free to reach out with questions, feedback, or contributions!

