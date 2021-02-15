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
from sklearn.model_selection import RandomizedSearchCV

# Can be useful to chose engine= 'xlrd'
df=pd.read_excel('data.xlsx')
df=df.drop(['country'],axis=1)
df=df[df['price']>0]
df.rename(columns={'statezip':'zip'}, inplace=True)
df['zip']=df['zip'].str.replace('WA','').astype(int)
df['floors']=df['floors'].astype(int)
#print(df)
y=df['price']
X=df.drop(['price'],axis=1)

max_cardinality = 10
high_cardinality = [col for col in X.select_dtypes(exclude=np.number)
                   if X[col].nunique() > max_cardinality]
X = X.drop(high_cardinality, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

# Models
DR = DummyRegressor(strategy='mean')
#RF = RandomForestRegressor(n_estimators=170,max_depth=25)
RF = RandomForestRegressor(n_estimators=800, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', max_depth=50, bootstrap=True)
DT = DecisionTreeRegressor(max_depth=25)
GB = GradientBoostingRegressor(learning_rate=0.01,n_estimators=200,max_depth=5)
LR = LinearRegression(n_jobs=10, normalize=True)

#{'n_estimators': 800, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': True}

models = [DR, RF, DT, GB, LR]
df_models = pd.DataFrame()
temp = {}

# Print the parameters of RF
print(RF.get_params())

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

#print(df.describe())

# Random hyperparameter Grid for RandomForest
# On each iteration, the algorithm will choose a
# difference combination of the features.
# 2 * 12 * 2 * 3 * 3 * 10 = 4320 settings in total.

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Random Search Training
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
#rf_random.fit(X_train, y_train)
#print(rf_random.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)