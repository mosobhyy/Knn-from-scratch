from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from KNN import KNN

"""# **Load iris Dataset**"""

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# print("X_train.shape: ", X_train.shape)
# print("y_train.shape: ", y_train.shape)
# print("X_test.shape: ", X_test.shape)
# print("y_test.shape: ", y_test.shape)

knn = KNN(k=3)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = knn.accuracy(predictions, y_test)

print(accuracy*100, '%')
