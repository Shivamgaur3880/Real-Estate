from sklearn.model_selection import cross_val_score

score= cross_val_score(model,strat_num_tr,housing_label,scoring="neg_mean_squared_error",cv=10)

rmse_score = np.sqrt(-score)
rmse_score

