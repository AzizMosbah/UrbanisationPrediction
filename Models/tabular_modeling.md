---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Modeling on tabular data 
    1. Looking at how a simple model does on the raw table in comparison to our enhanced table
    2. Check variable importance and change features accordingly
    3. Choosing a model and Hyperparameter tuning


#### Setting working directory to import the data 

```python
import os
os.chdir('../')
```

### Importing the packages the packages we need
    - Our pipelines to pull and transform the data to an 'ML-friendly' form
    - Some components of the sklearn library to process the data, measure the models' performances and the actual models

```python
from Pipelines.transformations import table_for_model, split
from Pipelines.process_raw import process_analytical_table, process_target
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
```

### Importing the final form of the tabular data 

```python
df = table_for_model()
train, test = split(df) #split between has_target and ~has_target
```

```python
df.loc[df.urban_housing_units > 1,['urban_housing_units']].describe()
```

```python
train, test = split(df_)
```

### Setting a Benchmark 
    - Set a benchmark by training a simple random forest on the variables that were given to us at the begining
    - Train the same model on our own variables for comparison, and to confirm our hypotheses
    - Check feature importance using entropy from the random forest

```latex
Since we want to optimize on RMSLE, we predict: $\newline$
    $\hat{y} := log(target + 1) \newline$
Then, we optimize that prediction for MSE which will be equivalent to:$\newline$
    $\frac{1}{n}\sum_{i=1}^{n}(\hat{y_i}-log(target_i + 1))^2 \newline$  
Which is equiavalent to: $\newline$
    $MSLE(e^{\hat{yi}}-1, target) \newline$
Thus, in the final submission we will give for every tile: $\newline$
    $exp(prediction) - 1\newline$

    
```

```python
train.loc[:,['log_target']] = np.log(1+train['target'])
```

```python
initial_predictors = ['empty_area', 'water_area',
                      'urban_area', 'barren_area', 
                      'forest_area', 'shrubland_area',
                      'herbaceous_area','cultivated_area', 
                      'wetland_area', 'protected_area_tnc', 
                      'protected_area_taa', 'elevation', 
                      'elevation_sd','elevation_min', 
                      'elevation_max', 'degree_of_slope',
                      'degree_of_slope_sd', 'degree_of_slope_min', 'degree_of_slope_max']

enhanced_predictors = ['empty_area', 'water_area',
                       'urban_area', 'barren_area', 
                       'forest_area', 'shrubland_area',
                       'herbaceous_area', 'cultivated_area', 
                       'wetland_area', 'protected_area_tnc', 
                       'protected_area_taa', 'elevation', 
                       'elevation_sd', 'elevation_min', 
                       'elevation_max', 'degree_of_slope',
                       'degree_of_slope_sd', 'degree_of_slope_min', 
                       'degree_of_slope_max','has_target', 
                       'housing_units', 'urban_housing_units',
                       'urban_housing_units_rate','neighbor1', 
                       'neighbor2', 'neighbor3']

       
top_contenders = ['urban_area','housing_units', 
                  'urban_housing_units', 'urban_housing_units_rate',
                  'neighbor1', 'neighbor2', 
                  'neighbor3', 'cultivated_area']

```

```python
initial_train = train.loc[:, initial_predictors]
enhanced_train = train.loc[:, enhanced_predictors]
```

#### Default Random Forest on initial data
    We predict the log of the target but we also print the actual target to verify that results are similar

```python
X_train, X_test, y_train, y_test = train_test_split(initial_train, train['log_target'], test_size=0.33, random_state=42)
_, _, _, y_target = train_test_split(initial_train, train['target'], test_size=0.33, random_state=42)
reg = RandomForestRegressor(n_estimators=100,max_depth=8, random_state=0, criterion = 'mse')
reg.fit(X_train, y_train)
```

```python
print("This model has an RMSE on the validation set of {0}".format(np.sqrt(mse(y_test, reg.predict(X_test)))))
```

```python
print("The RMSLE -when taking the exponential of the prediction minus one- is {0}".format(np.sqrt(msle(y_target, np.exp(reg.predict(X_test))-1))))
```

```python
importances = reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in reg.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, initial_predictors[indices[f]], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
```

#### Default Random Forest on enhanced data

```python
X_train, X_test, y_train, y_test = train_test_split(enhanced_train, train['log_target'], test_size=0.33, random_state=42)
_, _, _, y_target = train_test_split(enhanced_train, train['target'], test_size=0.33, random_state=42)
reg = RandomForestRegressor(n_estimators=100,max_depth=8, random_state=0, criterion = 'mse')
reg.fit(X_train, y_train)
```

```python
print("This model has an RMSE on the validation set of {0}".format(np.sqrt(mse(y_test, reg.predict(X_test)))))
```

```python
importances = reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in reg.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, enhanced_predictors[indices[f]], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
```

#### Now let us rerun the same benchmark model with the most predictive features


```python
top_contenders = ['neighbor1', 'neighbor2', 
                  'neighbor3', 'urban_area', 
                  'cultivated_area', 'urban_housing_units_rate',
                  'herbaceous_area', 'urban_housing_units',
                  'housing_units', 'forest_area', 'barren_area']
top_contenders_train = train.loc[:, top_contenders] 
```

```python
X_train, X_test, y_train, y_test = train_test_split(top_contenders_train, train['log_target'], test_size=0.1, random_state=42)
```

```python
reg = RandomForestRegressor(n_estimators=100,max_depth=6, random_state=0, criterion = 'mse')
```

```python
reg.fit(X_train, y_train)
```

```python
print("RMSE on the validation set: {0}".format(np.sqrt(mse(y_train, reg.predict(X_train)))))
print("RMSE on the test set: {0}".format(np.sqrt(mse(y_test, reg.predict(X_test)))))
```

### Nearest neighbors hypothesis. 
    - Two similar landscapes could have similar outcomes

```python
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=10)
```

```python
neigh.fit(X_train, y_train)
```

```python
print("RMSE on the validation set: {0}".format(np.sqrt(mse(y_train, neigh.predict(X_train)))))
print("RMSE on the test set: {0}".format(np.sqrt(mse(y_test, neigh.predict(X_test)))))
```

### Grid Search cross validation for Random Forest

```python
param_grid = {
    'bootstrap': [True],
    'max_depth': [2,4,6,8],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100]
}
rf = RandomForestRegressor()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
```

```python
print("RMSE on the validation set: {0}".format(np.sqrt(mse(y_train, grid_search.predict(X_train)))))
print("RMSE on the test set: {0}".format(np.sqrt(mse(y_test, grid_search.predict(X_test)))))
```

### Gradient Boosting Machine

```python
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
```

```python
print("RMSE on the validation set: {0}".format(np.sqrt(mse(y_train, reg.predict(X_train)))))
print("RMSE on the test set: {0}".format(np.sqrt(mse(y_test, reg.predict(X_test)))))
```

#### Let's try grid search on this one as well

```python
param_grid={'n_estimators':[200], 
            'learning_rate': [0.05],
            'max_depth':[4], 
            'min_samples_leaf':[3], 
            'criterion': ['mse'],
            'max_features': ['sqrt'],
            'init': ['zero']
            
           } 
gb = GradientBoostingRegressor()
grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
```

```python
print("RMSE on the validation set: {0}".format(np.sqrt(mse(y_train, grid_search.predict(X_train)))))
print("RMSE on the test set: {0}".format(np.sqrt(mse(y_test, grid_search.predict(X_test)))))
```

```python
grid_search.best_params_
```

### XGBoost

```python
xgbr = xgb.XGBRegressor(n_estimators =100,learning_rate = 0.01, max_depth = 8) 
xgbr.fit(X_train, y_train)
print("RMSE on the validation set: {0}".format(np.sqrt(mse(y_train, xgbr.predict(X_train)))))
print("RMSE on the test set: {0}".format(np.sqrt(mse(y_test, xgbr.predict(X_test)))))
```
