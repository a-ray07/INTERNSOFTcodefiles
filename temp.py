#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt 

#READING THE DATA FROM FILES
a=pd.read_csv('advertising.csv')
a.head()
 
#TO VISUALISE DATA
fig , axs = plt.subplots(1,3,sharey = True)
a.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
a.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
a.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

#CREATING X&Y FOR LINEAR REGRESSION
feature_cols=['TV']
x=a[feature_cols]
y=a.Sales

#IMPORTING LINEAR REGRESSION ALGO
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
print(lr.intercept_)
print(lr.coef_)

res=6.97+0.0554*50
print(res)

#CREATE A DATAFRAME WITH MIN & MAX VALUE OF THE TABLE
x_new=pd.DataFrame({'TV':[a.TV.min(),a.TV.max()]})
x_new.head()

p=lr.predict(x_new)
print(p)

a.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,p,c='red',linewidth=3)

import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales ~ TV',data=a).fit()
lm.conf_int()

#FINDING THE PROBABILITY VALUES
lm.pvalues

#FINDING THE R-SQUARED VALUES
lm.rsquared

feature_cols=['TV','Radio','Newspaper']
x=a[feature_cols]
y=a.Sales
lr=LinearRegression()
lr.fit(x,y)
print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=a).fit()
lm.conf_int()
lm.summary()


