corr_matrix=housing.corr() 

corr_matrix['MEDV'].sort_values(ascending=False) 

from pandas.plotting import scatter_matrix

attribute = ['RM','ZN','MEDV','LSTAT']
scatter_matrix(housing[attribute],figsize=(5,8))
plt.show()

housing.plot(kind="scatter",x='MEDV',y='LSTAT',alpha=1)      # df=pd.DataFrame()
plt.show()                                                   