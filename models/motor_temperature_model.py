"""
Motor Current and Temperature Model Architecture.

Implements:
  1. Current-Only Baseline (Classification + Anomaly Detection)
  2. Temperature-Only Baseline (Classification + Anomaly Detection)
  3. Combined Current-Plus-Temperature Model (Early Feature Fusion)

Uses scikit-learn classifiers and anomaly detectors with zero data leakage:
all scalers and transformations are fitted exclusively on training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ModelMetrics:
    """Evaluation metrics for classification and anomaly detection."""

    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    class_report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    anomaly_roc_auc: Optional[float] = None
    classes: List[str] = field(default_factory=list)


class MotorPdMModel:
    """Modular model supporting Current-Only, Temperature-Only, or Combined mode."""

    def __init__(
        self,
        mode: str = "combined",  # "current_only" | "temperature_only" | "combined"
        clf_type: str = "rf",    # "rf" | "gb" | "svm"
        random_state: int = 42,
    ):
        if mode not in ("current_only", "temperature_only", "combined"):
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode
        self.clf_type = clf_type
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.classifier: Optional[Any] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.classes_: List[str] = []
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False
        self.anomaly_threshold_: float = 0.5

    def _select_features(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Filter feature columns according to selected model mode."""
        if self.mode == "combined":
            return X, feature_names

        is_temp = np.array([f.startswith("temp_") for f in feature_names])
        if self.mode == "temperature_only":
            indices = np.where(is_temp)[0]
        else:  # "current_only"
            indices = np.where(~is_temp)[0]

        sub_names = [feature_names[i] for i in indices]
        return X[:, indices], sub_names

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
    ) -> "MotorPdMModel":
        """Fit scaler, classifier, and anomaly detector on training data only."""
        X_sub, self.feature_names_ = self._select_features(X_train, feature_names)

        # Fit scaler on training features
        X_scaled = self.scaler.fit_transform(X_sub)

        # Unique classes
        self.classes_ = sorted(list(set(y_train)))

        # Build and fit classifier
        if self.clf_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.clf_type == "gb":
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
            )
        elif self.clf_type == "svm":
            self.classifier = SVC(
                kernel="rbf",
                probability=True,
                C=10.0,
                gamma="scale",
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.clf_type}")

        self.classifier.fit(X_scaled, y_train)

        # Fit Isolation Forest anomaly detector on Normal instances in training set
        is_normal = np.array([y == "Normal" for y in y_train])
        if np.any(is_normal):
            X_healthy = X_scaled[is_normal]
        else:
            X_healthy = X_scaled

        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=self.random_state,
        )
        self.anomaly_detector.fit(X_healthy)

        # Calibrate anomaly threshold on training healthy set
        # score_samples returns opposite of anomaly score (higher = normal, lower = abnormal)
        raw_scores = -self.anomaly_detector.score_samples(X_healthy)
        self.anomaly_threshold_ = float(np.percentile(raw_scores, 95))
        self.anomaly_scale_ = max(0.02, float(np.std(raw_scores)))

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Predict condition class."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_sub, _ = self._select_features(X, feature_names)
        X_scaled = self.scaler.transform(X_sub)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Predict class posterior probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_sub, _ = self._select_features(X, feature_names)
        X_scaled = self.scaler.transform(X_sub)
        return self.classifier.predict_proba(X_scaled)

    def score_anomaly(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Compute calibrated anomaly scores in range [0.0, 1.0].

        Higher score denotes higher anomaly likelihood.
        """
        if not self.is_fitted or self.anomaly_detector is None:
            raise RuntimeError("Model is not fitted.")
        X_sub, _ = self._select_features(X, feature_names)
        X_scaled = self.scaler.transform(X_sub)

        # Raw score: higher means more anomalous
        raw = -self.anomaly_detector.score_samples(X_scaled)
        # Calibrate using sigmoid centered at threshold
        scale = getattr(self, "anomaly_scale_", None)
        if scale is None or scale <= 0:
            scale = max(0.02, float(np.std(raw)) if len(raw) > 1 else 0.05)
        calibrated = 1.0 / (1.0 + np.exp(-(raw - self.anomaly_threshold_) / scale))
        return np.clip(calibrated, 0.0, 1.0)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> ModelMetrics:
        """Evaluate model performance returning comprehensive classification & anomaly metrics."""
        y_pred = self.predict(X_test, feature_names)
        labels = self.classes_

        from sklearn.metrics import confusion_matrix

        acc = float(np.mean(y_pred == y_test))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

        # Anomaly detection ROC-AUC (binary: Normal = 0, Faulty = 1)
        anomaly_scores = self.score_anomaly(X_test, feature_names)
        is_fault = (y_test != "Normal").astype(int)
        roc_auc: Optional[float] = None
        if len(np.unique(is_fault)) > 1:
            try:
                roc_auc = float(roc_auc_score(is_fault, anomaly_scores))
            except Exception:
                pass

        return ModelMetrics(
            accuracy=acc,
            macro_f1=f1,
            macro_precision=prec,
            macro_recall=rec,
            class_report=rep,
            confusion_matrix=cm,
            anomaly_roc_auc=roc_auc,
            classes=labels,
        )
