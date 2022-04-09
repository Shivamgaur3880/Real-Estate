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


#DATA CORRELATION

corr_matrix=housing.corr() 

corr_matrix['MEDV'].sort_values(ascending=False) 

from pandas.plotting import scatter_matrix

attribute = ['RM','ZN','MEDV','LSTAT']
scatter_matrix(housing[attribute],figsize=(5,8))
plt.show()

housing.plot(kind="scatter",x='MEDV',y='LSTAT',alpha=1)      # df=pd.DataFrame()
plt.show()  


# TRAIN TEST SPLIT

from sklearn.model_selection import StratifiedShuffleSplit
split1 = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=43)

for train_index,test_index in split1.split(housing,housing['CHAS']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

    print(strat_train_set['CHAS'].value_counts())

    print(strat_test_set['CHAS'].value_counts())



# DATA CLEANING

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
imp.fit(strat_train_set)

print(imp.statistics_.shape)

print(imp.statistics_)

X=imp.transform(strat_train_set)
strat_train_set_tr=pd.DataFrame(X,columns=strat_train_set.columns)

print(strat_train_set_tr.describe())

#create features and label
housing_features= strat_train_set.drop('MEDV',axis=1)
housing_label=    strat_train_set["MEDV"]


# PIPELINE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([('imp1',SimpleImputer(strategy="median")),('std_scaler',StandardScaler())])

strat_num_tr = my_pipeline.fit_transform(housing_features)
print(strat_num_tr.shape)


# SELECTING A DESIRE MODEL

# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model= LinearRegression()
# model=DecisionTreeRegressor()
model = RandomForestRegressor()

model.fit(strat_num_tr,housing_label)

# DATA PREDICTION DEMO

some_data = housing_features.iloc[:5]
some_label = housing_label[:5]

prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

some_label

# PERFORMANCE MEASUREMENT

from sklearn.metrics import mean_squared_error

housing_predictions = model.predict(strat_num_tr)

lin_mse = mean_squared_error(housing_label,housing_predictions)
lin_rmse= np.sqrt(lin_mse)

print(lin_rmse)


# CROSS VALIDATION

from sklearn.model_selection import cross_val_score

score= cross_val_score(model,strat_num_tr,housing_label,scoring="neg_mean_squared_error",cv=10)

rmse_score = np.sqrt(-score)
print(rmse_score)


# PREPARE TEST DATA


strat_test_set.describe()

x_test = strat_test_set.drop("MEDV",axis=1)
y_test_label = strat_test_set["MEDV"]
x_test_prepaired = my_pipeline.transform(x_test)


final_prediction = model.predict(x_test_prepaired)
final_mse = mean_squared_error(y_test_label,final_prediction)
final_rmse = np.sqrt(final_mse)
print(final_prediction)
print("\n",list(y_test_label))

print(f"\nMean is {final_rmse.mean()}\n\n Standard Deviation is {final_rmse.std()}\n\n")


# MODEL Saving

from joblib import dump,load
dump(model,"Real_Estate.joblib")