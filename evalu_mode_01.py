# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 23:06:27 2021

@author: Caroline
"""

import pandas as pd
import numpy as np

# importar datos en limpio
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)

df.to_csv('module_5_auto.csv')

df=df._get_numeric_data()
df.head()

print(df.head())