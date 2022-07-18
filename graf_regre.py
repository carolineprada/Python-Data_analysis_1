# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 23:08:11 2021

@author: Caroline
"""

#Grafica
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# ruta de los datos
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


width = 12
height = 10
#plt.figure(figsize=(width, height))
#sns.regplot(x="highway-mpg", y="price", data=df)
#plt.ylim(0,)


plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

