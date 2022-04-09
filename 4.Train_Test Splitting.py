from sklearn.model_selection import StratifiedShuffleSplit
split1 = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=43)

for train_index,test_index in split1.split(housing,housing['CHAS']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

    strat_train_set['CHAS'].value_counts()

    strat_test_set['CHAS'].value_counts()

    