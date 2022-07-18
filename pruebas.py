# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 23:27:51 2021

@author: Caroline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats


path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


df_test = df[['body-style', 'price']]
df_grp = df_test.groupby(['body-style'], as_index=False).mean()

print(df_grp)


f=np.polyfit(x,y,3)
p=np.poly1d(f)