#!/usr/bin/python

#####################################################
# THIS IS IMPORTANT TO STOP THE WORLD BURNING DOWN! #
#####################################################
import matplotlib
matplotlib.use('TkAgg')
#####################################################

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier


features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################

clf = AdaBoostClassifier(n_estimators=100, random_state=0)
#clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(features_train, labels_train)

print 'Accuracy: {}'.format(clf.score(features_test, labels_test))

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
