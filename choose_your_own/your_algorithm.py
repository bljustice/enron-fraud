#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

clf1 = AdaBoostClassifier(n_estimators=50)
clf1.fit(features_train,labels_train)
pred1 = clf1.predict(features_test)
acc1 = accuracy_score(pred1,labels_test)
print 'AdaBoost accuracy' + str(acc1)

clf2 = SVC(kernel='rbf',C=20000)
clf2.fit(features_train,labels_train)
pred2 = clf2.predict(features_test)
acc2 = accuracy_score(pred2,labels_test)
print 'SVM accuracy:' + str(acc2)

clf3 = GaussianNB()
clf3.fit(features_train,labels_train)
pred3= clf3.predict(features_test)
acc3= accuracy_score(pred3,labels_test)
print 'Naive Bayes accuracy' + str(acc3)

clf4 = tree.DecisionTreeClassifier(min_samples_split=40)
clf4.fit(features_train,labels_train)
pred4 = clf4.predict(features_test)
acc4 = accuracy_score(pred4,labels_test)
print 'Decision Tree accuracy' + str(acc4)











try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
