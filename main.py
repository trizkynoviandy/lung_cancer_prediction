import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def select_dataset(dataset_name):
    if dataset_name == "Iris":
        df = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        df = datasets.load_breast_cancer()
    else:
        df = datasets.load_wine()
    X = df.data
    y = df.target

    return X, y

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        k = st.sidebar.slider('K', 1, 15)
        params["K"] = k
    elif clf_name == "SVM":
        c = st.sidebar.slider('C', 0.01, 10.0)
        params["C"] = c
    elif clf_name == "Naive Bayesian":
        pass
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimators = st.sidebar.slider("n Estimator", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Naive Bayesian":
        clf = GaussianNB()
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
    return clf

st.title('Data Classification by Machine Learning Algorithms')

dataset_selection = st.sidebar.selectbox("Select Datasets", ("Iris", 'Breast Cancer', 'Wine'))
classifier_selection = st.sidebar.selectbox("Select Classifier", 
('KNN', 'SVM', 'Random Forest', 'Naive Bayesian'))

X, y = select_dataset(dataset_selection)
st.write('Shape of the dataset', X.shape)
st.write('Number of classes', len(np.unique(y)))

params = add_parameter_ui(classifier_selection)
clf = get_classifier(classifier_selection, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

st.write(f"Accuracy Score on Training Set = {train_acc}")
st.write(f"Accuracy Score on Testing Set = {test_acc}")

# Visualize the data

pca = PCA(2)
X_train_projected = pca.fit_transform(X_train)
X_test_projected = pca.fit_transform(X_test)

x1_train = X_train_projected[:, 0]
x2_train = X_train_projected[:, 1]

x1_test = X_test_projected[:, 0]
x2_test = X_test_projected[:, 1]

fig=plt.figure(figsize=(12,5))
ax1=plt.subplot(1,2,1)
ax1.scatter(x1_train, x2_train, c=y_train, alpha=0.8, cmap='viridis')
ax1.set_title('Observed')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')

ax2=plt.subplot(1,2,2)
ax2.scatter(x1_train, x2_train, c=y_train_pred, alpha=0.8, cmap='viridis')
ax2.set_title('Predicted')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
st.pyplot(fig)

fig=plt.figure(figsize=(12,5))
ax1=plt.subplot(1,2,1)
ax1.scatter(x1_test, x2_test, c=y_test, alpha=0.8, cmap='viridis')
ax1.set_title('Observed')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')

ax2=plt.subplot(1,2,2)
ax2.scatter(x1_test, x2_test, c=y_test_pred, alpha=0.8, cmap='viridis')
ax2.set_title('Predicted')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
st.pyplot(fig)