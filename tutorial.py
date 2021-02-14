# Load libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import matplotlib.pyplot as plt

# Can be useful to chose engine= 'xlrd'
df=pd.read_excel('data.xlsx')
df=df.drop(['country'],axis=1)
df=df[df['price']>0]
df.rename(columns={'statezip':'zip'}, inplace=True)
df['zip']=df['zip'].str.replace('WA','').astype(int)
df['floors']=df['floors'].astype(int)
print(df)
y=df['price']
X=df.drop(['price'],axis=1)

max_cardinality = 10
high_cardinality = [col for col in X.select_dtypes(exclude=np.number)
                   if X[col].nunique() > max_cardinality]
X = X.drop(high_cardinality, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

models = [DummyRegressor(strategy='mean'),
          RandomForestRegressor(n_estimators=170,max_depth=25),
          DecisionTreeRegressor(max_depth=25),
          GradientBoostingRegressor(learning_rate=0.01,n_estimators=200,max_depth=5),
          LinearRegression(n_jobs=10, normalize=True)]
df_models = pd.DataFrame()
temp = {}

#run through models
for model in models:
    print(model)
    m = str(model)
    temp['Model'] = m[:m.index('(')]
    model.fit(X_train, y_train)
    temp['RMSE_Price'] = sqrt(mse(y_test, model.predict(X_test)))
    print('RMSE score',temp['RMSE_Price'])
    df_models = df_models.append([temp])
df_models.set_index('Model', inplace=True)

fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
df_models.RMSE_Price.plot(ax=axes, kind='bar', title='Root Mean Squared Error')
plt.show()

print(df.describe())