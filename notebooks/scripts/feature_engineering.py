
import pandas as pd


def create_features(train, test, target):
    # age bracks, 30s, 40s
    train = train.copy()
    test = test.copy()
    
    train["age"] = train["age"]//10
    test["age"] = test["age"]//10


    NUM_COL = []
    CAT_COL = []

    for col in test.columns:
        if test[col].dtype == object:
            CAT_COL.append(col)
        else:
            NUM_COL.append(col)

    # get_dummy_column 
    dummy_train = pd.get_dummies(train[CAT_COL])
    dummy_test = pd.get_dummies(test[CAT_COL])

    train_df = pd.concat([dummy_train, train[NUM_COL]], axis=1).fillna(-1)
    test_df = pd.concat([dummy_test, test[NUM_COL]], axis=1).fillna(-1)
    
    return train_df, test_df
    