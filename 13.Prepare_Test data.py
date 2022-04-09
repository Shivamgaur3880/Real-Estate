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