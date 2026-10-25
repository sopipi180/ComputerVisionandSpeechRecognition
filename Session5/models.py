import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 0


def model_zoo(seed=SEED):
    """
    {name: (estimator, needs_scaling)}.

    needs_scaling says whether the features must be standardised first.
    Distance- and weight-based models care (kNN, logistic, SVM); trees split one
    feature at a time, so a change of units cannot affect them.
    """
    return {
        "majority":  (DummyClassifier(strategy="most_frequent"), False),
        "logistic":  (LogisticRegression(max_iter=3000, class_weight="balanced",
                                         random_state=seed), True),
        "knn5":      (KNeighborsClassifier(n_neighbors=5, weights="distance"), True),
        "knn15":     (KNeighborsClassifier(n_neighbors=15, weights="distance"), True),
        "tree":      (DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                             random_state=seed), False),
        "forest":    (RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                             class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=seed), False),
        "svm_rbf":   (SVC(kernel="rbf", class_weight="balanced", cache_size=500,
                          random_state=seed), True),
    }


def family(name):
    """knn5 and knn15 are the same model with a different k - tune them once."""
    return "knn" if name.startswith("knn") else name


def param_grids():
    """
    Small hand-chosen grids, searched on the VALIDATION split.

    Deliberately short: a huge grid searched against one validation set mostly
    finds the setting that happens to suit those 786 lesions, which is a slower
    way of overfitting. Each list covers the settings that actually change how
    the model behaves.
    """
    knn = [{"model__n_neighbors": k, "model__weights": w}
           for k in (3, 5, 10, 25, 50) for w in ("uniform", "distance")]
    return {
        "logistic": [{"model__C": c} for c in (0.05, 0.2, 1.0, 5.0)],
        "knn": knn,
        "tree": [{"model__max_depth": d, "model__min_samples_leaf": leaf}
                 for d in (4, 8, 12) for leaf in (1, 5, 20)],
        "forest": [{"model__max_depth": d, "model__min_samples_leaf": leaf,
                    "model__max_features": f}
                   for d in (None, 12) for leaf in (1, 2, 5)
                   for f in ("sqrt", 0.3)],
        "svm_rbf": [{"model__C": c, "model__gamma": g}
                    for c in (0.5, 1.0, 5.0, 20.0) for g in ("scale", 0.05)],
    }


def build(estimator, needs_scaling):
    """Wrap the model so that standardising is part of it and cannot leak."""
    steps = ([("scale", StandardScaler())] if needs_scaling else []) + [("model", estimator)]
    return Pipeline(steps)


def scores(pipeline, X):
    """
    A score per class, for averaging the two photos of a lesion.

    predict_proba where the model has it; SVC without probability=True does not,
    so its signed distance to the boundary is used instead. Either way a larger
    value means "more likely this class", which is all the averaging needs.
    """
    model = pipeline[-1]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)
    d = pipeline.decision_function(X)
    return d if d.ndim == 2 else np.column_stack([-d, d])


#  scoring
def evaluate(y_true, y_pred, labels):
    """One row of results: the headline numbers plus recall for each class."""
    row = {
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for label, r in zip(labels, recall):
        row[f"recall_{label}"] = float(r)
    return row


def per_class_table(y_true, y_pred, labels):
    """Precision / recall / F1 / support per class, as the slides lay it out."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                        index=labels).round(3)


def confusion(y_true, y_pred, labels):
    """Rows = truth, columns = prediction."""
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                        index=[f"true {c}" for c in labels],
                        columns=[f"pred {c}" for c in labels])


# ------------------------------------------------- one prediction per lesion


def aggregate_to_lesions(score_rows, lesion_ids, classes):
    """
    Session 3's rule: a doctor decides about a LESION, not about a photo.

    Averages the two photos' class scores and takes the winner, so each lesion
    gets exactly one prediction.
    """
    df = pd.DataFrame(score_rows, columns=list(classes))
    df["lesion_id"] = np.asarray(lesion_ids)
    mean_scores = df.groupby("lesion_id").mean()
    return mean_scores.idxmax(axis=1)