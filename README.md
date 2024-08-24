# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### Developed by: Keerthi Vasan A
### Register No: 212222240048
### Date:
### AIM:
To Implement Linear and Polynomial Trend Estimation Using Python.

### ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using the least square method
4. Calculate the polynomial trend values using the least square method
5. End the program
### PROGRAM:
#### A - LINEAR TREND ESTIMATION
```py
Name : Keerthi Vasan A
Register No: 212222240048
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
df = pd.read_csv('/content/apple_stock.csv')
df.head(5)

# Convert 'Date' column to datetime format and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Convert 'Date' to ordinal format to use as numerical input for regression models
df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())
X = df['Date_ordinal'].values.reshape(-1, 1)
y = df['Adj Close'].values

# A - Linear Trend Estimation
linear_model = LinearRegression()
linear_model.fit(X, y)
df['Linear_Trend'] = linear_model.predict(X)

# Plotting the Linear Trend
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Adj Close'], label='Original Data', color='blue')
plt.plot(df['Date'], df['Linear_Trend'], color='yellow', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# B - Polynomial Trend Estimation
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
df['Polynomial_Trend'] = poly_model.predict(X_poly)

# Plotting the Polynomial Trend
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Adj Close'], label='Original Data', color='blue')
plt.plot(df['Date'], df['Polynomial_Trend'], color='green', label='Polynomial Trend (Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

```
### Dataset:

![Screenshot 2024-08-24 224430](https://github.com/user-attachments/assets/48f0f202-c0d8-4bc1-8698-c8a7d290cb5f)

### OUTPUT
A - LINEAR TREND ESTIMATION

![Screenshot 2024-08-24 224052](https://github.com/user-attachments/assets/fed8d7dd-7be2-4ca4-824d-a2bb7fcbf28a)



B- POLYNOMIAL TREND ESTIMATION

![Screenshot 2024-08-24 224105](https://github.com/user-attachments/assets/14fc7e2a-443f-4c9c-abd0-0dfe7e0a850f)

### RESULT:
Thus the Python program for linear and Polynomial Trend Estimation has been executed successfully.

