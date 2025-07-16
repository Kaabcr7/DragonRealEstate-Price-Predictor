Dragon Real Estate - Price Predictor
Project Overview
This project implements a machine learning model to predict housing prices, leveraging a dataset that includes various socio-economic factors and property attributes. The core of this predictor is a Random Forest Regressor, a robust ensemble learning algorithm known for its accuracy and ability to handle complex relationships within data.

The primary objective is to build a reliable predictive model that can estimate property sale prices, which can be useful for real estate valuation, market analysis, or informing investment strategies.

Features
Data Loading & Initial Exploration: Loads the housing dataset and performs initial data inspection (head, info, describe, value counts).

Data Visualization: Utilizes scatter plots and a scatter matrix to visualize relationships and correlations between features.

Attribute Combination: Explores creating new features (e.g., TAXRM) by combining existing ones to potentially improve model performance.

Train-Test Splitting: Implements stratified shuffling to ensure the training and testing datasets are representative of the overall data distribution, particularly for skewed categorical features like CHAS (Charles River dummy variable).

Missing Value Imputation: Handles missing numerical values using SimpleImputer with a median strategy.

Feature Scaling: Applies StandardScaler to normalize feature scales, crucial for many machine learning algorithms.

Pipeline Integration: A scikit-learn pipeline is used to streamline preprocessing steps (imputation and scaling), ensuring consistent transformations for both training and new data.

Random Forest Regressor: A powerful ensemble model is chosen for its predictive capabilities in regression tasks.

Model Evaluation:

Calculates Root Mean Squared Error (RMSE) on the training data.

Employs K-Fold Cross-Validation (10 folds) to provide a more robust and generalized estimate of the model's performance.

Model Persistence: The trained model and preprocessing pipeline are saved using joblib for future use without needing to retrain.

Prediction Module: Demonstrates how to load the saved model and make predictions on new, unseen data.

Technologies Used
Python 3.x

Pandas: For efficient data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib: For data visualization.

Scikit-learn: For machine learning algorithms, preprocessing tools, model selection, and pipelines.

Joblib: For saving and loading Python objects (models and pipelines).

Dataset
This project utilizes a housing price dataset, which is expected to be named data.csv and placed in the project's root directory. The dataset's columns are typically those found in the Boston Housing Dataset:

CRIM: Per capita crime rate by town

ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.

INDUS: Proportion of non-retail business acres per town

CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)

NOX: Nitric oxides concentration (parts per 10 million)

RM: Average number of rooms per dwelling

AGE: Proportion of owner-occupied units built prior to 1940

DIS: Weighted distances to five Boston employment centers

RAD: Index of accessibility to radial highways

TAX: Full-value property-tax rate per $10,000

PTRATIO: Pupil-teacher ratio by town

B: 1000(Bk−0.63) 
2
  where Bk is the proportion of Black residents by town

LSTAT: % lower status of the population

MEDV: Median value of owner-occupied homes in $1000s (Target variable)

Note: Ensure your data.csv file matches this column structure.

Installation and Setup
To set up and run this project locally, follow these steps:

Clone the Repository (if applicable, otherwise just create the directory):

git clone https://github.com/your-username/DragonRealEstate-Price-Predictor.git
cd DragonRealEstate-Price-Predictor

(Replace your-username with your actual GitHub username if cloning.)

Obtain data.csv: Download the Boston Housing Dataset (e.g., data.csv) and place it directly in the DragonRealEstate-Price-Predictor/ directory.

Create requirements.txt: Create a file named requirements.txt in the root of your project with the following content:

pandas
numpy
matplotlib
scikit-learn
joblib

Set Up a Virtual Environment (recommended):

python -m venv venv

Activate the Virtual Environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Install Jupyter Notebook (if you want to run the .ipynb file interactively):

pip install jupyter

How to Run the Project
The entire project workflow is encapsulated within the DragonRealEstate.ipynb Jupyter Notebook.

Launch Jupyter Notebook:

jupyter notebook

Your web browser will open, displaying the Jupyter interface.

Open the Notebook: Click on DragonRealEstate.ipynb to open it.

Execute Cells: Go through the notebook from top to bottom, executing each cell. The notebook will perform:

Data loading and initial exploration.

Train-test splitting.

Handling of missing values.

Feature scaling.

Model training (Random Forest Regressor).

Model evaluation using RMSE and 10-fold cross-validation.

Saving the trained model (Dragon.joblib).

Making predictions on a sample of the test data.

The outputs, including data descriptions, correlation matrices, model scores, and final predictions, will be displayed directly within the notebook.

Results
After running the notebook, you will see various outputs:

Initial Data Overview: Details on the dataset's structure, missing values, and descriptive statistics.

Correlations: Insights into how different features correlate with each other and with the target variable (MEDV).

Cross-Validation Scores: The mean and standard deviation of the RMSE scores from the 10-fold cross-validation, providing a robust measure of model performance. In your notebook, the rmse_scores mean is approximately 3.30.

Final Test Set RMSE: The Root Mean Squared Error of the model on the unseen test data. In your notebook, the final_rmse is approximately 2.91.

These metrics indicate how well the model predicts housing prices, with lower RMSE values signifying better accuracy.

Saving and Loading the Model
The trained RandomForestRegressor model is saved using joblib.

Saved Model File: Dragon.joblib

To load and use this model in a separate Python script or another notebook, you would typically load both the my_pipeline (for preprocessing) and the model.

Important Note on Saving: Your DragonRealEstate.ipynb currently saves only the model (dump(model, 'Dragon.joblib')). For a fully deployable system, it's best practice to save the entire fitted preprocessing pipeline as well.

To improve this for future deployments, consider modifying your notebook to save the my_pipeline object after it has been fit (e.g., after cell 39):

# In your DragonRealEstate.ipynb, after cell 39 (where my_pipeline is fit_transformed)
from joblib import dump
dump(my_pipeline, 'Dragon_pipeline.joblib')

And then, save your model as Dragon_model.joblib (or keep Dragon.joblib for the model).

If you save both, you can load them like this:

from joblib import load
import pandas as pd
import numpy as np # Needed for potential NaN values in new_data_example

# Load the saved pipeline and model
my_pipeline_loaded = load('Dragon_pipeline.joblib')
model_loaded = load('Dragon_model.joblib')

# Example of new data (this MUST mimic the structure of your original features DataFrame)
# Make sure the column names and data types are consistent with the training data.
# Use np.nan for any missing values, as your pipeline's imputer expects it.
new_data_example = pd.DataFrame({
    'CRIM': [0.00632], 'ZN': [18.0], 'INDUS': [2.31], 'CHAS': [0],
    'NOX': [0.538], 'RM': [6.575], 'AGE': [65.2], 'DIS': [4.0900],
    'RAD': [1], 'TAX': [296], 'PTRATIO': [15.3], 'B': [396.90], 'LSTAT': [4.98]
})

# Transform the new data using the loaded pipeline
processed_new_data = my_pipeline_loaded.transform(new_data_example)

# Make predictions using the loaded model
predicted_price = model_loaded.predict(processed_new_data)

print(f"Predicted price for the new data: ${predicted_price[0]:.2f} (in $1000s)")

Future Enhancements
Hyperparameter Tuning: Implement GridSearchCV or RandomizedSearchCV to systematically find the best hyperparameters for the RandomForestRegressor (or other models).

More Feature Engineering: Explore creating additional synthetic features that might capture more complex patterns in the data (e.g., polynomial features, interaction terms).

Outlier Detection & Handling: Analyze and potentially remove or transform outliers in the data that could skew model training.

Model Comparison: Systematically compare the performance of RandomForestRegressor with other models like LinearRegression, DecisionTreeRegressor, XGBoost, LightGBM, etc., using cross-validation.

Residual Analysis: Plot residuals (differences between actual and predicted values) to check for patterns that might indicate model weaknesses.

Deployment: Develop a simple web application (e.g., using Flask or FastAPI) to host the trained model as an API, allowing users to input property details and receive price predictions.

Interactive Visualizations: Create more interactive plots (e.g., using Plotly or Bokeh) for data exploration and presenting model results.

Contact
Your Name: [Your Name Here]

GitHub: [Link to your GitHub profile]

LinkedIn: [Link to your LinkedIn profile (Optional)]

Email: [Your Email Address (Optional)]

Feel free to reach out with any questions or feedback!
