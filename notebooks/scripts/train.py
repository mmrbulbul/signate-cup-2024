import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from datetime import datetime
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from data_cleaning import clean_data
from feature_engineering import create_features
from utils import seed_everything
from hyper_opt import objective_wrapper

import warnings
import yaml
warnings.filterwarnings('ignore')



SEED = 1971
seed_everything(SEED)

with open("../configs/config.yaml") as f:
    config = yaml.safe_load(f)
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%f")
# Start a W&B Run with wandb.init
run = wandb.init(project="signate_cup_2024",name=current_date, group="xgb", config=config)



# load data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# train.set_index("id", inplace=True)
# test.set_index("id", inplace=True)


# undersampling
# train = train.sample(frac=1).reset_index(drop=True)
# train_0 = train[train["ProdTaken"] == 0]
# train_1 = train[train["ProdTaken"] == 1]
# train = pd.concat([train_0.sample(len(train_1)*2), train_1])

# clean data 
train = clean_data(train)
test = clean_data(test)




target = train["prod_taken"]
train = train.drop(columns=["prod_taken"])



wandb_kwargs = {"project": "signate_cup_2024"}
wandbc = WeightsAndBiasesCallback(metric_name="auc_roc", wandb_kwargs=wandb_kwargs, as_multirun=False)

# model selection and validation 
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(study_name=current_date, direction='maximize', sampler=sampler)
objective = objective_wrapper(train, target, create_features, use_target_enc=False)
study.optimize(objective, callbacks=[wandbc], n_trials=30)


# log best hyperparameters

print('Best hyperparameters:', study.best_params)
print('Best auc:', study.best_value)

params = study.best_params
params.update({"objective": "binary:logistic",
        "verbosity": 0,
        "booster" : "gbtree",
        "eval_metric": "auc",
        "tree_method": 'exact',
        'random_state': SEED})

wandb.log({"model_params": params})




skf = StratifiedKFold(n_splits=5, random_state=1971, shuffle=True)



models = []
val_scores = []


for i, (train_index, valid_index) in enumerate(skf.split(train, target)):
    
    X_train, X_valid, y_train, y_valid = train.iloc[train_index], train.iloc[valid_index], target.iloc[train_index], target.iloc[valid_index]
    X_train, X_valid = create_features(X_train, X_valid, y_train, False)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False, eval_set=[(X_valid, y_valid)])
    models.append(model)
    
    train_pred = model.predict_proba(X_train)
    train_score = roc_auc_score(y_train, train_pred[:, 1])
    
    pred = model.predict_proba(X_valid)
    val_score = roc_auc_score(y_valid, pred[:, 1])

    val_scores.append(val_score)
    
    print(f"Fold {i}: train score {train_score} val score: {val_score}")
    
print("val score (avg)", np.mean(val_scores))

wandb.log({"VAL SCORE": np.mean(val_scores)})

feature_important = model.get_booster().get_score(importance_type='weight')

data = [[label, val] for (label, val) in feature_important.items()]
table = wandb.Table(data=data, columns=["feat", "importance"])
wandb.log(
    {
        "feat importance": wandb.plot.bar(
            table, "feat", "importance", title=" xgb feat importance"
        )
    }
)
# wandb.log({"feature_important": feature_important})
# submission
_, test_X = create_features(train, test, target, False)

weighted_val = val_scores /np.sum(val_scores)

predict = np.zeros((len(test_X), 2))

for model, weight in zip(models,weighted_val):
    predict += weight*model.predict_proba(test_X)

submit = pd.read_csv("../data/sample_submit.csv", header=None)

submit[1] = predict[:, 1]

outfile = f"../submission/{current_date}.csv"
submit.to_csv(outfile, index=False, header=None)
# wandb.save(outfile)

wandb.finish()



