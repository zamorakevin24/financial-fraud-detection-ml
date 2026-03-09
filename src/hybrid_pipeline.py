import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier


class HybridFraudPipeline:
    """
    Pipeline híbrido para detección de fraude:
    1. Genera anomaly_score con Isolation Forest
    2. Usa XGBoost para clasificar fraude
    """

    def __init__(
        self,
        contamination: float = 0.002,
        n_estimators_iso: int = 200,
        n_estimators_xgb: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators_iso = n_estimators_iso
        self.n_estimators_xgb = n_estimators_xgb
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.iso_model = None
        self.xgb_model = None
        self.feature_columns = None

    def _add_amount_log(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Amount_log"] = np.log1p(X["Amount"])
        return X

    def _add_anomaly_score(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["anomaly_score"] = self.iso_model.decision_function(X)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena todo el pipeline híbrido.
        """
        X = self._add_amount_log(X)

        # Entrenar Isolation Forest primero
        self.iso_model = IsolationForest(
            n_estimators=self.n_estimators_iso,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iso_model.fit(X)

        # Generar anomaly_score
        X_hybrid = self._add_anomaly_score(X)

        # Guardar orden de columnas
        self.feature_columns = list(X_hybrid.columns)

        # Balance de clases para XGBoost
        ratio = (y == 0).sum() / (y == 1).sum()

        # Entrenar XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=self.n_estimators_xgb,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=ratio,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss"
        )
        self.xgb_model.fit(X_hybrid, y)

        return self

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el mismo preprocessing del entrenamiento.
        """
        X = self._add_amount_log(X)
        X = self._add_anomaly_score(X)

        # Asegurar el mismo orden de columnas
        X = X[self.feature_columns]
        return X

    def predict(self, X: pd.DataFrame):
        X_prepared = self.prepare_features(X)
        return self.xgb_model.predict(X_prepared)

    def predict_proba(self, X: pd.DataFrame):
        X_prepared = self.prepare_features(X)
        return self.xgb_model.predict_proba(X_prepared)

    def feature_importances(self) -> pd.Series:
        """
        Devuelve importancias del modelo XGBoost.
        """
        return pd.Series(
            self.xgb_model.feature_importances_,
            index=self.feature_columns
        ).sort_values(ascending=False)