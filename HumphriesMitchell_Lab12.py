# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:00:28 2022

@author: hummitl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
iris = load_iris()




x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)


#Case 1
c1 = [100]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c1, random_state=1)
clf.fit(x_train, y_train)
print("Case 1:", clf.score(x_test, y_test))
#Case 2
c2 = [50,50]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c2, random_state=1)
clf.fit(x_train, y_train)
print("Case 2:", clf.score(x_test, y_test))
#Case 3
c3 = [75,25]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c3, random_state=1)
clf.fit(x_train, y_train)
print("Case 3:", clf.score(x_test, y_test))
#Case 4
c4 = [25,75]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c4, random_state=1)
clf.fit(x_train, y_train)
print("Case 4:", clf.score(x_test, y_test))
#Case 5
c5 = [75,25]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c5, random_state=1)
clf.fit(x_train, y_train)
print("Case 5:", clf.score(x_test, y_test))
#Case 6
c6 = [60,40]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c6, random_state=1)
clf.fit(x_train, y_train)
print("Case 6:", clf.score(x_test, y_test))
#Case 7
c7 = [50, 25]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c7, random_state=1)
clf.fit(x_train, y_train)
print("Case 7:", clf.score(x_test, y_test))
#Case 8
c8 = [25,25]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c8, random_state=1)
clf.fit(x_train, y_train)
print("Case 8:", clf.score(x_test, y_test))
#Case 9
c9 = [20,20]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c9, random_state=1)
clf.fit(x_train, y_train)
print("Case 9:", clf.score(x_test, y_test))
#Case 10
c10 = [30,70]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=c10, random_state=1)
clf.fit(x_train, y_train)
print("Case 10:", clf.score(x_test, y_test))