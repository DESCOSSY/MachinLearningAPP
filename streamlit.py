import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns

# from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

st.title('Project of Machine Learning.')

"Description of each column:"
"   - age - age in years"
"    - sex - (1 = male; 0 = female)"
"    - cp - chest pain type"
"    - trestbps - resting blood pressure (in mm Hg on admission to the hospital)"
"    - chol - serum cholestoral in mg/dl"
"    - fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)"
"    - restecg - resting electrocardiographic results"
"    - thalach - maximum heart rate achieved"
"    - exang - exercise induced angina (1 = yes; 0 = no)"
"    - oldpeak - ST depression induced by exercise relative to rest"
"    - slope - the slope of the peak exercise ST segment"
"    - ca - number of major vessels (0-3) colored by flourosopy"
"    - thal - 3 = normal; 6 = fixed defect; 7 = reversable defect"
"    - target - have disease or not (1=yes, 0=no)"

dataset = pd.read_csv('./heart.csv')
df1 = dataset.head(5)
st.write(df1)
"Shape of the data:"
st.write(dataset.shape)

dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

array = dataset.values
X = array[:, :-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)

	st.write('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# print(results)
# print(names)


# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# st.pyplot()

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
st.write("Test Accuracy : {:.2%}".format(accuracy_score(Y_validation, predictions)))