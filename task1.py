import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Example dataset
data = {
    'square_footage': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'bedrooms': [3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    'bathrooms': [2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
    'price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}
# Create a DataFrame
df = pd.DataFrame(data)
# Define features and target
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the linear regression model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)
# Print model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# Predict price for a new house
new_house = pd.DataFrame({'square_footage': [2500], 'bedrooms': [4], 'bathrooms': [3]})
predicted_price = model.predict(new_house)
print("Predicted price for the new house:", predicted_price)
