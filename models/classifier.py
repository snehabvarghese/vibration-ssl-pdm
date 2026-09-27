"""
Downstream classifiers that operate on FROZEN self-supervised embeddings.

WHY THEY ARE TINY
-----------------
The whole point of the self-supervised stage is that representation learning
already happened without labels. What is left for the labelled data to do is
only to draw decision boundaries in the 128-D embedding space, which needs far
fewer labels than learning a CNN from scratch. So the classifiers here are
deliberately lightweight:

  svm_rbf : SVC with an RBF kernel, on standardised embeddings. Strong with
            very few samples per class and has only two hyper-parameters.
  mlp     : one hidden layer (128 units). Slightly more flexible; also gives
            calibrated-ish probabilities for the dashboard's confidence value.

Both are wrapped in an sklearn Pipeline with a StandardScaler, fitted on the
labelled subset only -- the scaler must not see test statistics.

INPUT  : embeddings (N, 128) + integer labels
OUTPUT : fitted estimator with .predict and .predict_proba
"""

from __future__ import annotations

from typing import Optional

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CFG, DownstreamConfig


def build_classifier(
    name: str,
    cfg: Optional[DownstreamConfig] = None,
    seed: int = CFG.seed,
) -> Pipeline:
    """Factory for the downstream classifiers ('svm_rbf' | 'mlp')."""
    cfg = cfg or CFG.downstream
    if name == "svm_rbf":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,      # needed for the dashboard confidence
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(cfg.mlp_hidden,),
            activation="relu",
            solver="adam",
            learning_rate_init=cfg.mlp_lr,
            max_iter=cfg.mlp_epochs * 10,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier {name!r}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
