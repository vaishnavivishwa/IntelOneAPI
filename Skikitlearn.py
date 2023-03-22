from sklearnex import patch_sklearn
patch_sklearn()
# Digits Recognizer from Sklearn 8*8

from sklearn.datasets import load_digits
import time
start_time = time.time()
digits = load_digits()

# Dataframe
import pandas as pd
df = pd.DataFrame(digits.data)
df['Target'] = digits.target
df

x = digits.data # Input pixel values as features
y = digits.target

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

from sklearn.svm import SVC

model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_pred,y_test))

print(classification_report(y_pred,y_test))
print(f"It took {time.time() - start_time:.2f} seconds")

OutPut
==
0.9911111111111112
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        37
           1       1.00      0.98      0.99        44
           2       1.00      1.00      1.00        44
           3       0.98      1.00      0.99        44
           4       1.00      1.00      1.00        38
           5       0.98      0.98      0.98        48
           6       1.00      1.00      1.00        52
           7       1.00      1.00      1.00        48
           8       0.98      0.98      0.98        48
           9       0.98      0.98      0.98        47

    accuracy                           0.99       450
   macro avg       0.99      0.99      0.99       450
weighted avg       0.99      0.99      0.99       450

It took 0.10 seconds

# California Housing Dataset

import time
start_time = time.time()
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
# from sklearnex import unpatch_sklearn
# unpatch_sklearn()

# =============== dataset for the project =====================================

dataset = fetch_california_housing()


# ============== Polynomial Features for the dataset ==========================

pft = PolynomialFeatures(degree = 6)

# ============== Label names ==================================================

label_prices = dataset['target']
feature_names = dataset['feature_names']

# ============== Feature Normalization of the dataset =========================
data_original = (dataset.data)
X_scaled = preprocessing.scale(dataset.data)

# ================= Generating poly features ==================================

X_poly = pft.fit_transform(X_scaled)

# ================= Splitting the dataset(train, validation and test ==========
X_train, X_dummy, y_train, y_dummy = train_test_split(X_poly, dataset.target, test_size = 0.40, random_state = 42)
X_CV,X_test,y_CV,y_test = train_test_split(X_dummy, y_dummy, test_size = 0.2, random_state = 42)

# ================= Fit a linear regression model =============================
model = linear_model.Ridge(alpha = 9000)
model.fit(X_train, y_train)

predictionCV = model.predict(X_CV)
predictionTestSet = model.predict(X_test)

errorCV = mean_squared_error(y_CV, predictionCV)
errorTestSet = mean_squared_error(y_test, predictionTestSet)

print(f"It took {time.time() - start_time:.2f} seconds")

Output
==
It took 4.57 seconds

Dataset
==
{'data': array([[   8.3252    ,   41.        ,    6.98412698, ...,    2.55555556,
           37.88      , -122.23      ],
        [   8.3014    ,   21.        ,    6.23813708, ...,    2.10984183,
           37.86      , -122.22      ],
        [   7.2574    ,   52.        ,    8.28813559, ...,    2.80225989,
           37.85      , -122.24      ],
        ...,
        [   1.7       ,   17.        ,    5.20554273, ...,    2.3256351 ,
           39.43      , -121.22      ],
        [   1.8672    ,   18.        ,    5.32951289, ...,    2.12320917,
           39.43      , -121.32      ],
        [   2.3886    ,   16.        ,    5.25471698, ...,    2.61698113,
           39.37      , -121.24      ]]),
 'target': array([4.526, 3.585, 3.521, ..., 0.923, 0.847, 0.894]),
 'frame': None,
 'target_names': ['MedHouseVal'],
 'feature_names': ['MedInc',
  'HouseAge',
  'AveRooms',
  'AveBedrms',
  'Population',
  'AveOccup',
  'Latitude',
  'Longitude'],
 'DESCR': '.. _california_housing_dataset:\n\nCalifornia Housing dataset\n--------------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 20640\n\n    :Number of Attributes: 8 numeric, predictive attributes and the target\n\n    :Attribute Information:\n        - MedInc        median income in block group\n        - HouseAge      median house age in block group\n        - AveRooms      average number of rooms per household\n        - AveBedrms     average number of bedrooms per household\n        - Population    block group population\n        - AveOccup      average number of household members\n        - Latitude      block group latitude\n        - Longitude     block group longitude\n\n    :Missing Attribute Values: None\n\nThis dataset was obtained from the StatLib repository.\nhttps://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html\n\nThe target variable is the median house value for California districts,\nexpressed in hundreds of thousands of dollars ($100,000).\n\nThis dataset was derived from the 1990 U.S. census, using one row per census\nblock group. A block group is the smallest geographical unit for which the U.S.\nCensus Bureau publishes sample data (a block group typically has a population\nof 600 to 3,000 people).\n\nAn household is a group of people residing within a home. Since the average\nnumber of rooms and bedrooms in this dataset are provided per household, these\ncolumns may take surpinsingly large values for block groups with few households\nand many empty houses, such as vacation resorts.\n\nIt can be downloaded/loaded using the\n:func:`sklearn.datasets.fetch_california_housing` function.\n\n.. topic:: References\n\n    - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,\n      Statistics and Probability Letters, 33 (1997) 291-297\n'}
