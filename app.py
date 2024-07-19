# Import Libraries
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Loading Dataset
iris = load_iris()

# X and Y for Training
X = iris.data
Y = iris.target

# Split
x_train, x_test, Y_train, Y_tests = train_test_split(X, Y, test_size=0.3, random_state=0)

clf = GaussianNB()
clf.fit(x_train, Y_train)

y_pred = clf.predict(x_test)
print(accuracy_score(Y_tests, y_pred))

with open("model.pkl", "wb") as file:
    pickle.dump(clf, file)

with open("model.pkl", "rb") as file:
    data = pickle.load(file)

print(data.predict(x_test))
print(data.score(x_test, Y_tests))
