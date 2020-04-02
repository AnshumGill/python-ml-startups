import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import scipy.stats as stats
import statsmodels.api as sm

dataset=pd.read_csv("50_Startups.csv")
data=dataset.drop(['Administration','State'],axis=1)
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

'''
This models R-sqaured is 0.950
Adj. R-sqaured is 0.948
Std deviation 0.15
'''
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_pred=reg.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=reg,X=X_train,y=Y_train,cv=10)
accuracies.mean()
accuracies.std()

x=sm.add_constant(X)
regressor_OLS=sm.OLS(endog=Y,exog=x).fit()
regressor_OLS.summary()

'''
Taking Average
This Models R-squared is 0.735
and Adj. R-sqaured is 0.730
Std deviation 1.5
'''
new_X=pd.DataFrame()
new_X['Avg']=(X['R&D Spend']+X['Marketing Spend'])/2

X_train,X_test,Y_train,Y_test=train_test_split(new_X,Y,test_size=0.2)
new_reg=LinearRegression()
new_reg.fit(X_train,Y_train)
y_pred=new_reg.predict(X_test)
new_accuracies=cross_val_score(estimator=new_reg,X=X_train,y=Y_train,cv=10)
new_accuracies.mean()
new_accuracies.std()

x=sm.add_constant(new_X)
new_regressor_OLS=sm.OLS(endog=Y,exog=x).fit()
new_regressor_OLS.summary()

