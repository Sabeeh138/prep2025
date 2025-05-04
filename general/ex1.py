from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1,2,3,4,5]).reshape(-1, 1)
y = np.array([2,3,4,5,6])

model = LinearRegression()
model.fit(x, y)
print("Prediction: ", model.predict(x))