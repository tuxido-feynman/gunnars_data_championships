import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from subprocess import check_output
from sklearn import metrics, model_selection
# /Users/mli3/ml/Kaggle/train.csv

def load_data(csv_path):
    return pd.read_csv(csv_path)

# load the data
csv_path = '/Users/mli3/ml/Kaggle/'
train = load_data(csv_path + "train.csv")
test = load_data(csv_path + "test.csv")


train.describe()

# remove columns where the value is the same for all rows (no new information)
nunique = train.apply(pd.Series.nunique)
col_same_train = list(nunique[nunique==1].index)
train2 = train.drop(col_same_train,1)

# apply the same to the test set
test2 = test.drop(col_same_train, 1)
# remove columns where the column is same as another column

# hist of Y

import matplotlib.pyplot as plt
y_train = train2[["y"]]
y_train.hist(bins=200, figsize=(20,15))
plt.show()

# remove row without the maximum y (looks like outliet)
train3 = train2.loc[data['y']!=data['y'].max()]

c = train3.corr()
s = c.unstack()
so = s.sort_values(kind='quicksort', ascending=False)
so['y']

# remove columns that are identical and also those that are complete opposite
col_same = s[((s==1) | (s==-1)) & (s.index.get_level_values(0) != s.index.get_level_values(1))]
#keep the smaller of the columns
# if x is smaller than the other columns, throw other columns out, else ignore
dup_columns = set()
for x,y in col_same.index:
    if int(x[1:]) < int(y[1:]):
        dup_columns.add(y)

# there are 45+8 dup_columns
train4 = train3.drop(list(dup_columns), 1)
test4 = test2.drop(list(dup_columns), 1)
# impute missing values

# # this part might be unecessary
# from sklearn.preprocessing import Imputer

# imputer = Imputer(strategy="median")

# df_dedup_num = df_dedup.drop(["ID", "y", "X0",  "X1",  "X2", "X3", "X4", "X5", "X6", "X8"], axis=1)
# imputer.fit(df_dedup_num)
# imputer.statistics_
# df_dedup_num.median().values
# X = imputer.transform(df_dedup_num)
# df_dedup_num_tr = pd.DataFrame(X, columns=df_dedup_num.columns)

# 1 hot encoding


train_num = train4.drop(["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"], 1) # kept y column
train_cat = train4.loc[:, ["ID", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]]



test_num = test4.drop(["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"], 1)
test_cat = test4.loc[:, ["ID", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]]
#from sklearn.preprocessing import LabelBinarizer
#encoder = LabelBinarizer()
#data_dedup_cat_1hot = encoder.fit_transform(data_dedup_cat)
#data_dedup_cat_1hot


train_one_hot_cat = pd.get_dummies(train_cat, columns=["X0", "X1",  "X2", "X3", "X4", "X5", "X6", "X8"])
train_one_hot = train_one_hot_cat.merge(train_num, on='ID')

test_one_hot_cat = pd.get_dummies(test_cat, columns=["X0", "X1",  "X2", "X3", "X4", "X5", "X6", "X8"])
test_one_hot = test_one_hot_cat.merge(test_num, on='ID')

X = train_one_hot.drop(['ID', 'y'], axis=1)
y = train_one_hot['y']


feature_labels = X.columns
forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X, y)    # Fits the Random Forest Regressor to the entire data set.
importances = forest.feature_importances_  # Sets importances equal to the feature importances of the model
indices = np.argsort(importances)[::-1]
order_features = []
order_importances = []
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feature_labels[indices[f]], importances[indices[f]]))
    order_features.append(feature_labels[f])
    order_importances.append(importances[indices[f]])





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score



test = test_one_hot.drop(['IDID'], axis=1).drop(order_features[200:], axis=1)
train = train_one_hot.drop(['IDID'], axis=1).drop(order_features[200:], axis=1) # Modify train to only take in the top 100 features and the target column y




# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train.drop(['y'],axis=1), train[['y']])


forest_scores = cross_val_score(forest_reg, train.drop(['y'],axis=1), train[['y']], 
    scoring="neg_mean_squared_error", cv=10)

forest_rmse = np.sqrt(-forest_scores)
display_scores(forest_rmse)


y_pred = fores_reg.predict(test.drop(['y'], axis=1))













# Custom Transformer
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# num_pipeline = Pipeline([
#    ('imputer', Imputer(strategy="median"),
#        ('attribs_adder', CombinedAttributesAdder()),
#        ('std_scaler', StandardScaler()),
#        ])

# Pipeline
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
    ])
    
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape

    




