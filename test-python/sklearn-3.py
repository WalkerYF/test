from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
# data
data_X = loaded_data.data
# label
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

# print(model.predict(data_X[:4, :]))
# print(data_y[:4])
# print(model.coef_)
# print(model.intercept_)
# print(model.get_params())
print(model.score(data_X, data_y)) # R^2 coefficient of determination