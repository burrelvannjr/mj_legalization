python2

import scipy
import numpy
import matplotlib
import pandas
import sklearn
import random, sys, csv
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dataset = pandas.read_csv('st_2018.csv')

print(dataset.shape)

print(dataset.describe())

array = dataset.values
X = array[:,2:5]
Y = array[:,1]
Y = Y.astype('int')
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=3, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


fig = plt.figure()
fig.suptitle('Comparing Algorithms')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
samp = [[1,1,5],[1,0,0]]
predictions = knn.predict(samp)
predictions



sample = numpy.array(samp)
df = sample.T
df2 = numpy.array([predictions])
df3 = numpy.append(df, df2, axis=0)


pred = df3.T

csvfile = "df.csv"


with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(["dem","initiative","mjballots","prediction"])
    writer.writerows(pred)


















