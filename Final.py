# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:22:00 2021

@author: Caroline
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge




#Se exporta el archivo
file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
df.head()


##Dividiremos los datos entre conjunto de entrenamiento y conjunto de prueba:
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])



##PARTE 1
#Tipo de datos
df.dtypes
print(df.dtypes)



#Método, resultados estadísticos
print(df.describe())


##PARTE 2
#Elimine las columnas "id" y "Unnamed: 0" del eje 1 mediante el método drop(), después utilice el método 
#describe() para obtener un resumén estadístico de los datos. Tome una impresión de pantalla y envíela, 
#asegurese de que el parámetro inplace sea True

df.drop("Unnamed: 0", axis = 1, inplace = True)
df.drop("id", axis = 1, inplace = True)

print(df.describe())


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


##PARTE 3
#Utilice el método value_counts para contabilizar el número de casa con un solo piso, use el método .to_frame() 
#para convertirlo en un dataframe.

df['floors'].value_counts().to_frame()

##PARTE 4
#Utilice la función boxplot de la librería seaborn para determinar si las casas cuando tienen 
#o no vista al mar presentan precios atípicos.

#sns.boxplot(x="waterfront", y="price", data=df)


##PARTE 5
#Utilice la función regplot de la librería seaborn para determinar si la característica 
#sqft_above esta relacionada con el precio negativa o positivamente.
sns.regplot(x="sqft_above", y="price", data=df, ci=None)


print(df.corr()['price'].sort_values())


##PARTE 6
#Podemos ajustar un modelo de regresión lineal utilizando la característica de longitud 'long' y calcular R^2.
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

#Ajuste un modelo de regresión lineal para predecir 'price' utilizando la característica 'sqft_living' 
#y despues calcule R^2. Tome una impresión de pantalla de su código y del valor de R^2.

X1 = df[['sqft_living']]
Y1 = df['price']
lm = LinearRegression()
lm
lm.fit(X1,Y1)
lm.score(X1, Y1)



##PARTE 7
#Ajuste un modelo de regresión lineal para predecir 'price' utilizando la lista de características (features):
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     

##Después calcule R^2. Tome una impresión de pantalla de su código.
X2 = df[features]
Y2 = df['price']
lm.fit(X2,Y2)
lm.score(X2,Y2)



##PARTE 8
#Genere una lista de tuplas, el primer elemento de la tupla contiene el nombre del estimador:
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

#Utilice la lista para crear un objeto de tipo pipeline para predecir 'price', ajuste el objeto 
#utilizando las características en la lista features y calcule R^2.
pipe=Pipeline(Input)
print(pipe)


##Dividiremos los datos entre conjunto de entrenamiento y conjunto de prueba:
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


##PARTE 9
#Genere y ajuste un objeto de regresión sesgada utilizando los datos de entrenamiento, 
#establezca el parámetro de regularización a 0.1 y calcule R^2 usando los datos de prueba.
RidgeModel = Ridge(alpha=0.1) 
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)


##PARTE 10
#Realice una transformación polinómica de segundo grado en los conjuntos de entrenamiento y prueba. 
#Genere y ajuste un objeto de regresión sesgada con los datos de entrenamiento, establezca el parametro 
#de regularización a 0.1 y calcule R^2 con los datos de prueba. Tome una impresión de pantalla de su 
#codigo y el valor de R^2.

pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[features])
x_test_pr=pr.fit_transform(x_test[features])

RigeModel = Ridge(alpha=0.1) 
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr, y_test)























