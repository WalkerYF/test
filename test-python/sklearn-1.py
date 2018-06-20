import numpy as np
from sklearn import datasets

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X[:2, :])
# print(iris_y)
print(iris.DESCR)

# 训练集和测试集
# 分开成两个集合，并且打乱数据
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

# print(X_test.shape)
# print(y_test.shape)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# print(knn.predict(X_test))
# print(y_test)