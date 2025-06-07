import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

try:
    from git import Repo
except Exception:
    Repo = None

LOG_PATH = Path('logs/trade_outcomes.csv')
MODEL_PATH = Path('ml_model/triangular_rf_model.pkl')


def load_trade_logs(flag: str = None) -> pd.DataFrame:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f'{LOG_PATH} not found')
    df = pd.read_csv(LOG_PATH)
    if flag:
        df = df[df['outcome_flag'] == flag]
    return df


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    features = ['confidence_entry', 'spread_zscore', 'cointegration_entry']
    df = df.dropna(subset=features + ['outcome_flag'])
    X = df[features]
    y = df['outcome_flag']
    if len(X) < 2:
        raise ValueError('Not enough data for training')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {acc:.4f}')
    print(f'Class distribution: {y.value_counts().to_dict()}')
    return clf


def save_model(model: RandomForestClassifier):
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')
    if Repo and Path('.git').exists():
        try:
            repo = Repo(Path('.'))
            repo.index.add([str(MODEL_PATH)])
            repo.index.commit(f'Retrain triangular_rf_model {datetime.utcnow().isoformat()}')
        except Exception as e:
            logging.error(f'Git commit failed: {e}')


def main():
    try:
        df = load_trade_logs('success')
        model = train_model(df)
        save_model(model)
    except Exception as e:
        logging.error(f'Retraining failed: {e}')


if __name__ == '__main__':
    main()

