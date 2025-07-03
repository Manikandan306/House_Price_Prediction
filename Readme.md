House Price Prediction Project
Overview
This is a beginner-level data science project that predicts house prices using a linear regression model. The project demonstrates fundamental skills in data preprocessing, model training, evaluation, and visualization using Python. It uses a simple dataset with features like square footage, number of bedrooms, and bathrooms to predict house prices.
Dataset
The dataset (house_prices.csv) contains 20 rows with the following columns:

square_feet: The size of the house in square feet.
bedrooms: Number of bedrooms.
bathrooms: Number of bathrooms.
price: The house price in USD (target variable).

Prerequisites

Python 3.8 or higher
Required libraries listed in requirements.txt

Installation

Clone or download this project to your local machine.
Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:pip install -r requirements.txt



Usage

Ensure house_prices.csv is in the same directory as main.py.
Run the main script:python main.py


The script will:
Load and preprocess the dataset.
Train a linear regression model.
Evaluate the model using Mean Squared Error (MSE) and R-squared metrics.
Generate a plot (price_prediction_plot.png) comparing actual vs. predicted prices.
Output a sample prediction for a house with 2000 sq ft, 3 bedrooms, and 2 bathrooms.



Files

main.py: The main Python script for data processing, model training, and visualization.
house_prices.csv: Sample dataset for house price prediction.
requirements.txt: Lists the required Python libraries.
price_prediction_plot.png: Output plot showing actual vs. predicted prices (generated after running the script).

Results
The model provides:

Mean Squared Error (MSE) to measure prediction error.
R-squared score to indicate how well the model explains the variance in house prices.
A scatter plot visualizing actual vs. predicted prices.

Skills Demonstrated

Data preprocessing with Pandas
Linear regression modeling with Scikit-learn
Model evaluation using MSE and R-squared
Data visualization with Matplotlib

Future Improvements

Add more features (e.g., location, age of the house) to improve model accuracy.
Experiment with other algorithms (e.g., Random Forest, XGBoost).
Include cross-validation for robust model evaluation.
