from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
imp.fit(strat_train_set)

imp.statistics_.shape

imp.statistics_

X=imp.transform(strat_train_set)
strat_train_set_tr=pd.DataFrame(X,columns=strat_train_set.columns)

strat_train_set_tr.describe()

#create features and label
housing_features= strat_train_set.drop('MEDV',axis=1)
housing_label=    strat_train_set["MEDV"]

