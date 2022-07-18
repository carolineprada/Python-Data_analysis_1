# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 22:45:57 2021

@author: Caroline
"""
#Regresión Simple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# ruta de los datos
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


lm = LinearRegression()

#Variables
X = df[['highway-mpg']]
Y = df['price']

#Entrena el modelo
lm.fit(X,Y)

Yhat = lm.predict(X)
print(Yhat[0:5]  )

lm.intercept_
lm.coef_

