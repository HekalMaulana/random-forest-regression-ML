import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Training random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=0, n_estimators=10)
regressor.fit(X,y)

# Predicting the result
y_predict = regressor.predict([[6]])
print(y_predict)

# Visualing the random forest regression with high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth Or Bluff (Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

