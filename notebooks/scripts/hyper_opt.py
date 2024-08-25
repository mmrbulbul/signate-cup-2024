
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import wandb
import random

def objective_wrapper(X, y, func, seed=43, use_target_enc=False):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
            "verbosity": 0,
            "booster" : "gbtree",
            "eval_metric": "auc",
            "tree_method": 'exact',
            'random_state': seed,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }

        scores = []
        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
            train_index = random.choices(train_index, k=len(valid_index)) 
            X_train, X_valid, y_train, y_valid = X.iloc[train_index], X.iloc[valid_index], y.iloc[train_index], y.iloc[valid_index]

            X_train, X_valid = func(X_train, X_valid, y_train, use_target_enc)
        
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False, eval_set=[(X_valid, y_valid)])
            predictions = model.predict_proba(X_valid)
            auc_score = roc_auc_score(y_valid, predictions[:,1])
            scores.append(auc_score)
            wandb.log({"trial_no": trial.number, "cv_no": i+1,  "auc_score": auc_score})
        wandb.log({"trial_no": trial.number ,  "mean_auc_score": np.mean(scores)})
        return np.mean(scores)
    return objective