"""
This script can be used as skeleton code to read the challenge train and test
geojsons, to train a trivial model, and write data to the submission file.
"""
# data handling
import geopandas as gpd
import pandas as pd
import numpy as np

# data analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score

change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
                   'Mega Projects': 5}

# Read csvs
print("--- read .csv files ---")
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

######## Feature engineering ########
print("--- Feature engineering ---")


def get_features(df):
    features = []

    # geometry features
    perimeter = np.asarray(df['geometry'].length)
    perimeter = np.expand_dims(perimeter, axis=-1)
    features.append(perimeter)

    area_values = np.asarray(df['geometry'].area)
    area_values = np.expand_dims(area_values, axis=-1)
    features.append(area_values)

    # geography features
    le_urban_type = LabelEncoder()
    le_urban_type.fit(np.asarray(df["urban_type"]))
    print("possible urban_type list :", list(le_urban_type.classes_))
    urban_type = np.asarray(df["urban_type"])
    urban_type = le_urban_type.transform(urban_type)
    urban_type = np.expand_dims(urban_type, axis=-1)
    features.append(urban_type)

    le_geography_type = LabelEncoder()
    le_geography_type.fit(np.asarray(df["urban_type"]))
    print("possible geography_type list :", list(le_geography_type.classes_))
    geography_type = np.asarray(df["urban_type"])
    geography_type = le_urban_type.transform(geography_type)
    geography_type = np.expand_dims(geography_type, axis=-1)
    features.append(geography_type)

    for feat in features:
        print(feat.shape)

    res = np.concatenate(features, axis=-1)

    return res


train_x = get_features(train_df)
train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

test_x = get_features(test_df)

print("train_x.shape, train_y.shape, test_x.shape :\n",
      train_x.shape, train_y.shape, test_x.shape)


######## Training ########
print("--- train ---")
knn_clf = KNeighborsClassifier(n_neighbors=3)
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
    estimators=[('knn', knn_clf), ('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')

print('!!! mean f1 score on -training- set !!!')
for clf in (knn_clf, log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(train_x, train_y)
    pred_y = clf.predict(train_x)
    print(clf.__class__.__name__, f1_score(pred_y, train_y))

voting_clf.fit(train_x, train_y)
pred_y = voting_clf.predict(train_x)
print("f1_score on training set :", f1_score(pred_y, train_y))

pred_y = voting_clf.predict(test_x)
print("prediction on test set shape :", pred_y.shape)

######## Save results to submission file ########
print("--- save ---")
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("my_submission.csv", index=True, index_label='Id')
