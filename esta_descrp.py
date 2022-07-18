# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:11:19 2021

@author: Caroline
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()
#print(df.dtypes)

df.corr()

#Correlación
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


#REGRESION LINEAL
##Gráfico de dispersión

#sns.regplot(x='engine-size', y='price', data= df)
#sns.regplot(x='compression-ratio', y='horsepower', data= df)
#plt.ylim(0, )

#Correlación entre tamaño motor y precio
#df[["engine-size", "price"]].corr()
#sns.regplot(x="highway-mpg", y="price", data=df)


#Correlación revoluciones vs. precio
#sns.regplot(x="peak-rpm", y="price", data=df)
df[['peak-rpm','price']].corr()


#Correlación
df[["stroke","price"]].corr()
#sns.regplot(x="stroke", y="price", data=df)


#VARIABLES CATEGORICAS
#Gráfico de cajas y bigotes


#sns.boxplot(x="body-style", y="price", data=df)
#sns.boxplot(x="engine-location", y="price", data=df)
#sns.boxplot(x="drive-wheels", y="price", data=df)

#ANALISIS ESTADISTICO
df.describe()

df.describe(include=['object'])
df['drive-wheels'].value_counts()
df['drive-wheels'].value_counts().to_frame()



#Cambiar nombre a la columna
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)



#AGRUPACION
#Categorías de agrupamiento
df['drive-wheels'].unique()

df_group_one = df[['drive-wheels','body-style','price']]

df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()


df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()


#Pivot 
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')

#Completa valores faltantes con cero (0)
grouped_pivot = grouped_pivot.fillna(0)

#Valor promedio
df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
#print(grouped_test_bodystyle)

## MAPA DE CALOR
#plt.pcolor(grouped_pivot, cmap='RdBu')
#plt.colorbar()
#plt.show()





fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# nombres de las etiquetas 
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# mover etiquetas al centro 
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# insertar etiquetas 
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotar las etiquetas si son muy largas 
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()













