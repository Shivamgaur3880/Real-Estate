from sklearn.metrics import mean_squared_error

housing_predictions = model.predict(strat_num_tr)

lin_mse = mean_squared_error(housing_label,housing_predictions)
lin_rmse= np.sqrt(lin_mse)

lin_rmse

