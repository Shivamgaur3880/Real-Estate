housing['TAXRM']=housing['TAX']/housing['RM']

corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing.plot(kind='scatter',x='TAXRM',y='MEDV')
plt.show()

strat_train_set.describe()