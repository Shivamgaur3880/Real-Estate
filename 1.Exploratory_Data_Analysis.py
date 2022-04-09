import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

housing=pd.read_csv("data.csv")

housing.head()       # shows first 5 rows

housing.info()      # we check missing data

housing['CHAS']          # shows CHAS data   

housing['CHAS'].value_counts()      # shows no.of values

housing.describe()  # describe data

housing.hist(bins=30,figsize=(20,15))     # bins is bars width and figsize is figure size
plt.show()  