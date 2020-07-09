#!/usr/bin/python
# importing libraries
import pandas as pd
from sklearn import datasets
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

def user_input():
	sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
	sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
	petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
	petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
	data = {'sepal_length': sepal_length,
			'sepal_width': sepal_width,
			'petal_length': petal_length,
			'petal_width': petal_width}
	features = pd.DataFrame(data, index=[0])
	return features


if __name__ == '__main__':
	st.write("""
		# Simple Iris Flower Prediction App
		This app predicts the **Iris flower** type!
		""")

	st.sidebar.header('User Input Parameters')
	st.subheader('User Input parameters')
	
	df = user_input()
	st.write(df)

    # Load dataset
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target
    
    # Applied K-nearest Neighbours algorithms
	knn = KNeighborsClassifier(n_neighbors=5)
	# Model fitting 
	knn.fit(X,Y)
	prediction = knn.predict(df)

	st.subheader('Class labels and their corresponding index number')
	st.write(iris.target_names)

	st.subheader('Prediction')
	st.write(iris.target_names[prediction])