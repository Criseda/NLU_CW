import numpy as np
import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_lr(trial, X, y):
    C        = trial.suggest_float("C", 1e-4, 1e2, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    model = LogisticRegression(
        solver='saga', l1_ratio=l1_ratio, C=C,
        max_iter=5000, random_state=42,
    )
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    return scores.mean()


def objective_lgbm(trial, X, y):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 3000),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    return scores.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",  required=True, help="Path to full (N, 97) .npy feature matrix")
    parser.add_argument("--labels",    required=True)
    parser.add_argument("--model",     required=True, choices=["lr", "lgbm"])
    parser.add_argument("--n_trials",  type=int, default=500)
    parser.add_argument("--output",    required=True, help="Path to SQLite study .db")
    args = parser.parse_args()

    X_raw = np.load(args.features)
    y     = np.load(args.labels).astype(int)

    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)

    if args.model == "lr":
        X = StandardScaler().fit_transform(X)
        objective = lambda trial: objective_lr(trial, X, y)
        study_name = "logreg_tuning"
    else:
        objective = lambda trial: objective_lgbm(trial, X, y)
        study_name = "lgbm_tuning"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.output}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    print(f"Running {args.n_trials} Optuna trials for {args.model.upper()} ...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\nBest F1 macro: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save best params as text alongside the .db
    output_path = Path(args.output)
    with open(output_path.with_suffix(f".{args.model}.best_params.txt"), "w") as f:
        f.write(f"Best F1 macro: {study.best_value:.4f}\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")


if __name__ == "__main__":
    main()
