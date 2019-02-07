import pandas as pd
import numpy as np
import pickle

from sklearn import linear_model

#loading our data as a panda
df = pd.read_csv('winequality-red.csv', delimiter=";")
label = df['quality'] 
features = df.drop('quality', axis=1)

#defining our linear regression estimator and training it with our wine data
regr = linear_model.LinearRegression()
regr.fit(features, label)

#print(regr.predict([[7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4]]).tolist())
#serializing our model to a file called model.pkl
pickle.dump(regr, open("model.pkl", "wb"))

#loading a model from a file called model.pkl
model = pickle.load(open("model.pkl","rb"))