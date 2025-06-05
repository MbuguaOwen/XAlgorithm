import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils.filters import MLFilter
import tempfile


def test_mlfilter_predicts_with_sample_model(tmp_path):
    # create simple training data
    X = pd.DataFrame({
        'f1': [0, 1, 2],
        'f2': [0, 1, 2],
        'f3': [0, 1, 2]
    })
    y = [0, 1, 2]
    model = RandomForestClassifier(n_estimators=5, random_state=1)
    model.fit(X, y)

    model_path = tmp_path / 'model.pkl'
    joblib.dump(model, model_path)

    filt = MLFilter(str(model_path))
    test_features = pd.DataFrame({'f1': [2], 'f2': [2], 'f3': [2]})
    confidence, signal = filt.predict_with_confidence(test_features)

    assert 0.0 <= confidence <= 1.0
    assert signal == 1  # class 2 should map to signal 1


