# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:09:50 2021

@author: WDY
"""


import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score as acc_rate

# use this line in terminal if graphviz does not work: conda install python-graphviz 

p2p=pd.read_csv("p2p.csv",index_col=0)

# Glance the data set
print(p2p.columns.values)  # check the label
print(p2p.index)  # check the size

p2p_data=p2p.drop("status",axis=1)
p2p_target=p2p["status"]

# randomly choose 1/3 of samples as testing data
np.random.seed(123)
test_idx = np.random.randint(0, len(p2p_target), len(p2p_target) // 3)

# training data
train_data = np.delete(np.array(p2p_data), test_idx, axis=0)
train_target = np.delete(np.array(p2p_target), test_idx)

# testing data
test_data = np.array(p2p_data)[test_idx]
test_target = np.array(p2p_target)[test_idx]

# train the model
# initial an decision tree classifier object with given arguments
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  splitter='best')
# A lot of arguments can be placed into the object
# Refer docs: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
clf.fit(X=train_data, y=train_target)

# make prediction
print('\nThe target test data set is:\n', test_target)
print('\nThe predicted result is:\n', clf.predict(test_data))
print('\nAccuracy rate is:\n', acc_rate(test_target, clf.predict(test_data)))

# visualizing the tree
#print(list(p2p["status"]))
#print(list(p2p_data.columns.values))

dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names=list(p2p_data.columns.values),
                                class_names=list("status"),
                                filled=True,
                                rounded=True,
                                impurity=False,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("p2p", view = True)
