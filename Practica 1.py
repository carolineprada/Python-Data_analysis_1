# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:21:27 2021

@author: Caroline
"""

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import matplotlib as plt
from matplotlib import pyplot



filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


df = pd.read_csv(filename, names = headers)
df.head()

##Reemplaza los valores sin dato
df.replace("?", np.nan, inplace = True)
df.head(5)


##Evaluan datos faltantes
missing_data = df.isnull()
missing_data.head(5)



##Cuenta los valores faltantes
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


##Calcular el valor de la promedio → NORMALIZED-LOSSES
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

#Reemplazo de valores sin definir → NORMALIZED-LOSSES
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)



##Calcular el valor de la media → BORE
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

#Reemplazo de valores sin definir → BORE
df["bore"].replace(np.nan, avg_bore, inplace=True)



#PARTE 1


##Calcular el valor de la media → STROKE
avg_stroke = df['stroke'].astype('float').mean(axis=0)
print("Average Stroke:", avg_stroke)

##Reemplazar datos sin definir → STROKE
df["stroke"].replace(np.nan, avg_stroke, inplace=True)


##Calcular el valor de la media → HORSEPOWER
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

##Reemplazar datos sin definir → HORSEPOWER
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


##Calcular el valor de la media → PEAK-RPM
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

##Reemplazar datos sin definir → PEAK-RPM
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


##Contar datos
df['num-of-doors'].value_counts()


##Modelo carros más común
df['num-of-doors'].value_counts().idxmax()

##Reemplazan los valores en el número de puertas, por la moda entre la data
df["num-of-doors"].replace(np.nan, "four", inplace=True)


##Eliminar todos los datos que al final quedaron sin información
df.dropna(subset=["price"], axis=0, inplace=True)


##Se restablece el indice, ya que al eliminar datos, estos quedan saltados
df.reset_index(drop=True, inplace=True)

#Muestra los primeros 5 registros
df.head()

#Muestra el tipo de dato, de cada objeto
df.dtypes


#Convierten a otro tipo de dato
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

##Mostrar marco de datos, con los nuevos tipos de datos
df.dtypes


##ESTANDARIZACIÓN DE DATOS

# Convertir m.p.g. a L/100km mediante una operación matematica (235 dividido por m.p.g.)
df['city-L/100km'] = 235/df["city-mpg"]

df['highway-mpg'] = 235/df["highway-mpg"]

#Reemplazar
df.rename(columns={"highway-mpg":"highway-L/100km"}, inplace = True)



#Escalado simple → Length
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


#Escalado simple → Height
df['height'] = df['height']/df['height'].max() 
df[["length","width","height"]].head()



##DISCRETIZACIÓN

#Corregir formato
df["horsepower"]=df["horsepower"].astype(int, copy=True)


#Graficar en un histograma de los caballos de fuerza para ver la apariencia de su distribución.

plt.pyplot.hist(df["horsepower"])

##Establece las etiquetas x/y y muestra el título 
##Inicial
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


##Contenedores
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


##Nombre de los grupos de contenedores
group_names = ['Low', 'Medium', 'High']


##Función cut → Ordenar y Segmentar
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


##Conteo de las cantidades en los grupos
df["horsepower-binned"].value_counts()


##Distribuidor del contenedor
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

##Medio
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")



##Visualizador de contenedor
a = (0,1,2)

# Dibuja el histograma del atributo "horsepower" con bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

##
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

#Muestra indices
df.columns

##VARIABLES CATEGORICAS EN VARIABLES CUANTITATIVAS
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


##Cambiar valores
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()


df = pd.concat([df, dummy_variable_1], axis=1)

#eliminar la columna original "fuel-type" de "df" 
df.drop("fuel-type", axis = 1, inplace=True)


##Variable aspiration
dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

dummy_variable_2.head()


##Variable aspiration - parte 2
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop('aspiration', axis = 1, inplace=True)


#Guardar el archivo tranformado
df.to_csv('clean_df.csv')




avg=df['horsepower'].mean(axis=0)
print(df['horsepower'].replace(np.nan, avg))
