import geopandas

landmap = geopandas.read_file("/Users/mac/R/LCA-GB/Data/Input/LANDMAP.geojson")
landmap = landmap.dropna(subset=['ScenicQuality'])

print(landmap)
print(landmap.describe)
print(landmap['ScenicQuality'].value_counts())

landmap.isnull().sum()

landmap_A = landmap.query("Consultant in ['A']")
landmap_B = landmap.query("Consultant in ['B']")

print(landmap.shape)
print(landmap_A.shape)
print(landmap_B.shape)

def split_consultant(landmap):
  X = landmap.loc[:, ['Scenicness.mean', 'Scenicness.entropy', 'Wildness.median', 'Wildness.IQR']]
  y = landmap.loc[:, ['ScenicQuality']]
  from sklearn.preprocessing import OrdinalEncoder
  from sklearn.compose import make_column_transformer
  import pandas as pd

  # Instantiate the label encoder
  transformer = make_column_transformer(
      (OrdinalEncoder(categories=[['Low', 'Moderate', 'High', 'Outstanding']]), ['ScenicQuality'])
  )

  y = transformer.fit_transform(y)
  y = pd.DataFrame(y, columns = ['ScenicQuality'])
  return X, y

def EDA_plot(y, X):
  import pandas as pd
  # concatenating y and X along columns
  landmap_concat = pd.DataFrame(pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1))
  print(landmap_concat.shape)

  import seaborn as sns
  sns.set()
  sns.pairplot(landmap_concat, hue='ScenicQuality', palette='vlag', height=3)

# split train/test set
X, y = split_consultant(landmap_A)
print(y.shape, X.shape)
EDA_plot(y, X)

ratio = 0.8
from sklearn.model_selection import train_test_split
# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - ratio, train_size=ratio, random_state=19,
                                                    stratify=y)

import xgboost as xgb
# Specify sufficient boosting iterations to reach a minimum
num_round = 1000 #209.91

# Leave most parameters as default
param = {'objective': 'multi:softmax',  # Specify multiclass classification
         'num_class': 4,  # Number of possible output classes
         # 'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
         'eta': 0.451,
         'max_depth': 12,
         'gamma': 0,
         'subsample': 0.758,
         'colsample_bytree': 1,
         'min_child_weight': 0.2,
         'tree_method': 'hist',
         'predictor': 'cpu_predictor'
         }

# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

import time
gpu_res = {}  # Store accuracy result
start = time.time()

# Train model
# model = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
# print("GPU Training Time: %s seconds" % (str(time.time() - start)))
model = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')])
print(time.time() - start)

# Evaluate model using test data
from sklearn.metrics import accuracy_score
predictions = model.predict(dtest)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test, predictions)
print("Kappa: %.2f%%" % (kappa))

# save to JSON
model.save_model("/Users/mac/PycharmProjects/XGBoost/models/model_A.json")