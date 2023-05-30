import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
#housing=pd.read_csv("housing 1.csv",encoding = "ISO-8859-1",on_bad_lines='skip',lineterminator='\n')
housing=pd.read_csv("housing 1.csv")
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
#print(f"rows in train set:{len(train_set)}\nrows in test set:{len(test_set)}\n")
from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing[' CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

housing=strat_train_set.copy()
corr_matrix=housing.corr()
corr_matrix[' MEDV'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes=[" MEDV"," RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))
housing.plot(kind="scatter",x=" RM",y=" MEDV",alpha=0.8)
housing["TAX RM"]=housing['TAX']/housing[' RM']
corr_matrix=housing.corr()
corr_matrix[' MEDV'].sort_values(ascending=False)
housing.plot(kind="scatter",x="TAX RM",y=" MEDV",alpha=0.8)
housing=strat_train_set.drop(" MEDV",axis=1)
housing_labels=strat_train_set[" MEDV"].copy()
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)
x=imputer.transform(housing)
housing_tr=pd.DataFrame(x,columns=housing.columns)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([('imputer',SimpleImputer(strategy="median")),('std_scaler',StandardScaler()),])
housing_num_tr=my_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
#def print_scores(scores):
   # print("scores",scores)
  #  print("mean:",scores.mean())
 #   print("standard deviation:",scores.std())
#print_scores(rmse_scores)
from joblib import dump, load
dump(model,'dragon.joblib')
x_test=strat_test_set.drop(" MEDV",axis=1)
y_test=strat_test_set[" MEDV"].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_predictions=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
#print(final_predictions,list(y_test))
from joblib import dump, load
import numpy as np
model=load('dragon.joblib')
st.header("PROPERTY VALUE ESTIMATOR")
Lstate = st.number_input('Insert your Lstat value')
tax = st.number_input('Insert your property tax')
t_p = st.number_input('Insert your teacher to pupil ratio')
crim = st.number_input('Insert your crime rate per town')
features =np.array([[crim,  7.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24714252, -99.31238772,  2.61111401, -1.0016859 , tax ,
       t_p, 0.41164221, Lstate]])
y=int(model.predict(features)*80*1000)

st.write("PROPERTY VALUE")
st.write(y)


