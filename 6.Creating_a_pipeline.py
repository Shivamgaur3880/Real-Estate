from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([('imp1',SimpleImputer(strategy="median")),('std_scaler',StandardScaler())])

strat_num_tr = my_pipeline.fit_transform(housing_features)
strat_num_tr.shape

