import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Basic data exploration
print("Dataset Info:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Handle missing values (if any)
data = data.dropna()

# Define features (X) and target (y)
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.savefig('price_prediction_plot.png')
plt.show()

# Example prediction for a new house
new_house = np.array([[3000, 4, 4]])  # Example: 2000 sq ft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(new_house)
print(f"\nPredicted price for a house with 3000 sq ft, 4 bedrooms, 4 bathrooms: ${predicted_price[0]:.2f}")