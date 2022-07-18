# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 23:13:41 2021

@author: Caroline
"""

#EJERCICIOS
##Importar paquetes, librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures


#importar datos en limpio
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)


df.to_csv('module_5_auto.csv')

y_data = df['price']

x_data=df.drop('price',axis=1)


##Líneas para dividir el conjunto de datos al 15%
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


##Líneas para dividir el conjunto de datos al 40%
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0) 
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])


lre=LinearRegression()
#Se entrena el modelo
lre.fit(x_train[['horsepower']], y_train)

#Se calcula el R^2 con datos de prueba 
lre.score(x_test[['horsepower']], y_test)

#Se calcula el R^2 más pequeño
lre.score(x_train[['horsepower']], y_train)


##Líneas para dividir el conjunto de datos al 90%
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.9, random_state=0) 
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])



#Puntuación con validación cruzada
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

##R^2
Rcross
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())


##Error cuadratico
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

#Promedio del R^2
Rc=cross_val_score(lre,x_data[['horsepower']], y_data,cv=2)
Rc.mean()

yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]


##SOBREAJUSTE, SUBAJUSTE
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]

yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]



#Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
#DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


#Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
#DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)


##SOBREAJUSTE
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr


poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)
yhat[0:5]

print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)


PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)


poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)


Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')    


def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)

interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))













