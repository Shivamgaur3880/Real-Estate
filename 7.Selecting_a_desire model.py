# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model= LinearRegression()
# model=DecisionTreeRegressor()
model = RandomForestRegressor()

model.fit(strat_num_tr,housing_label)

