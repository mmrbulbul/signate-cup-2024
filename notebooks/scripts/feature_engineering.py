
import pandas as pd
import numpy as np
from encoders import TargetEncoder
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
    
import_col = ['designation', 'product_pitched', 'monthly_income', 'age',
       'marital_status', 'number_of_followups', 'pitch_satisfaction_score',
       'typeof_contact', 'duration_of_pitch', 'number_of_person_visiting']

def create_features(train, test, target, use_target_encoder=False, seed=1971):
    # age bracks, 30s, 40s
    train = train.copy()
    test = test.copy()
    
    # test = test[import_col]
    # train = train[import_col]
    
    # create odd age bracket 
    train["odd_age"]  = train["age"].apply(lambda x: 
                          'normal' if x >=18  and x >= 61  else 
                          'outliar')
    test["odd_age"]  = test["age"].apply(lambda x: 
                          'normal' if x >=18  and x >= 61  else 
                          'outliar')
    

    
    
    # income to age ratio
    train["income_by_age"] = train["monthly_income"]/train["age"]
    test["income_by_age"] = test["monthly_income"]/test["age"]
    
    
    # replace age with age brackets
    train["age"] = train["age"]//10
    test["age"] = test["age"]//10


    NUM_COL = []
    CAT_COL = []

    for col in test.columns:
        if test[col].dtype == object:
            CAT_COL.append(col)
        else:
            NUM_COL.append(col)
            
    # create columns based on none column:
    test_none = pd.DataFrame()
    train_none = pd.DataFrame()
    for col in train.columns:
        if train[col].isnull().any():
            train_none[f'{col}_has_none'] = train[col].isnull()
            test_none[f'{col}_has_none'] = test[col].isnull()
            

    # handle missing value
    # train[NUM_COL] = train[NUM_COL].fillna(train[NUM_COL].median())
    # test[NUM_COL] = test[NUM_COL].fillna(train[NUM_COL].median())
    
    # get_dummy_column 
    if use_target_encoder:
        te = TargetEncoder(categories=CAT_COL)
        dummy_train = te.fit_transform(train[CAT_COL], target)
        dummy_test = te.transform(test[CAT_COL])
    else:
        # one hot encoder
        dummy_train = pd.get_dummies(train[CAT_COL])
        dummy_test = pd.get_dummies(test[CAT_COL])
    
    # select columns for kmeans
    cols = [ 'marital_status', 'age']
    kmeans_col = []
    for col in cols:
        for col2 in dummy_train.columns:
            if col in col2:
                kmeans_col.append(col2)

    kmeans = KMeans(n_clusters=5, random_state=seed).fit(dummy_train[kmeans_col])
    train["Cluster"] = kmeans.fit_predict(dummy_train[kmeans_col])
    train["Cluster"] = train["Cluster"].astype("category")
    
    test["Cluster"] = kmeans.predict(dummy_test[kmeans_col])
    test["Cluster"] = test["Cluster"].astype("category")
    
    dummy_cluster_train = pd.get_dummies(train["Cluster"], prefix="Cluster")
    dummy_cluster_test = pd.get_dummies(test["Cluster"], prefix="Cluster")
    

    train_df = pd.concat([dummy_train, train[NUM_COL], dummy_cluster_train, train_none], axis=1)
    test_df = pd.concat([dummy_test, test[NUM_COL], dummy_cluster_test, test_none], axis=1)
    
    return train_df, test_df
    