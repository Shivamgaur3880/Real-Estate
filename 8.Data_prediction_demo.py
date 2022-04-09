some_data = housing_features.iloc[:5]
some_label = housing_label[:5]

prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

some_label

