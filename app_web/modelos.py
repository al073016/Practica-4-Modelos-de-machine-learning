"""Entrenamiento real e inferencia de las tres redes de la Práctica 4.

Las particiones, arquitecturas y criterios de selección reproducen los notebooks
del repositorio. Los datasets se descargan de sus fuentes públicas al entrenar.
"""

from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from typing import Callable
from urllib.request import urlopen

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
Report = Callable[[str, int, int, dict], None]


def _download(url: str) -> bytes:
    with urlopen(url, timeout=90) as response:
        return response.read()


@lru_cache(maxsize=1)
def bank_data() -> pd.DataFrame:
    outer = _download("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip")
    with zipfile.ZipFile(io.BytesIO(outer)) as archive:
        nested = archive.read("bank.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read("bank-full.csv")), sep=";")
    assert len(frame) == 45211 and "y" in frame
    return frame


@lru_cache(maxsize=1)
def energy_data() -> pd.DataFrame:
    blob = _download("https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = next(name for name in archive.namelist() if name.lower().endswith(".xlsx"))
        frame = pd.read_excel(io.BytesIO(archive.read(name)), engine="openpyxl")
    frame = frame.iloc[:, :10].copy()
    frame.columns = [f"X{i}" for i in range(1, 9)] + ["Y1", "Y2"]
    frame = frame.dropna(subset=["Y1"])
    assert len(frame) == 768
    return frame


@lru_cache(maxsize=1)
def forda_data() -> tuple[np.ndarray, np.ndarray]:
    root = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    train = np.loadtxt(io.BytesIO(_download(root + "FordA_TRAIN.tsv")), delimiter="\t", dtype="float32")
    test = np.loadtxt(io.BytesIO(_download(root + "FordA_TEST.tsv")), delimiter="\t", dtype="float32")
    assert train.shape == (3601, 501) and test.shape == (1320, 501)
    return train, test


def _callbacks(label: str, epochs: int, report: Report | None, patience: int, reduce_patience: int):
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if report is not None:
                report(label, epoch + 1, epochs, logs or {})

    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=reduce_patience, factor=0.5),
        Progress(),
    ]


def _losses(history) -> dict[str, list[float]]:
    return {"entrenamiento": [float(v) for v in history.history["loss"]],
            "validación": [float(v) for v in history.history["val_loss"]]}


def train_bank(report: Report | None = None) -> dict:
    df = bank_data()
    X = df.drop(columns=["y", "duration"])
    y = (df["y"] == "yes").astype(int).to_numpy()
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    X_train_p = prep.fit_transform(X_train).astype("float32")
    X_val_p = prep.transform(X_val).astype("float32")
    X_test_p = prep.transform(X_test).astype("float32")
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    def build(widths):
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(X_train_p.shape[1],))])
        for width in widths:
            model.add(tf.keras.layers.Dense(width, activation="relu"))
            model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
        return model

    trials = []
    for widths in ((32, 16), (64, 32)):
        tf.keras.utils.set_random_seed(SEED)
        model = build(widths)
        history = model.fit(X_train_p, y_train, validation_data=(X_val_p, y_val),
                            epochs=50, batch_size=128, class_weight=class_weight, verbose=0,
                            callbacks=_callbacks(str(widths), 50, report, 7, 3))
        p_val = model.predict(X_val_p, verbose=0).ravel()
        trials.append((roc_auc_score(y_val, p_val), widths, model, history, p_val))
    best_auc, widths, model, history, p_val = max(trials, key=lambda item: item[0])
    del trials
    thresholds = np.arange(0.05, 0.951, 0.01)
    threshold = float(thresholds[np.argmax([f1_score(y_val, p_val >= t) for t in thresholds])])
    p_test = model.predict(X_test_p, verbose=0).ravel()
    pred = (p_test >= threshold).astype(int)
    return {
        "kind": "bank", "model": model, "X_test": X_test, "X_test_p": X_test_p, "y_test": y_test,
        "threshold": threshold, "configuration": widths, "history": _losses(history),
        "metrics": {
            "ROC-AUC": float(roc_auc_score(y_test, p_test)),
            "PR-AUC": float(average_precision_score(y_test, p_test)),
            "F1 (sí contrata)": float(f1_score(y_test, pred)),
            "Precisión (sí contrata)": float(precision_score(y_test, pred)),
            "Recall (sí contrata)": float(recall_score(y_test, pred)),
        },
        "baseline": float(y_test.mean()), "confusion": confusion_matrix(y_test, pred),
        "sizes": (len(X_train), len(X_val), len(X_test)), "validation_score": float(best_auc),
    }


def train_energy(report: Report | None = None) -> dict:
    df = energy_data()
    X = df[[f"X{i}" for i in range(1, 9)]].to_numpy(dtype="float32")
    y = df["Y1"].to_numpy(dtype="float32").reshape(-1, 1)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=SEED)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_train_p = x_scaler.fit_transform(X_train).astype("float32")
    X_val_p = x_scaler.transform(X_val).astype("float32")
    X_test_p = x_scaler.transform(X_test).astype("float32")
    y_train_p = y_scaler.fit_transform(y_train).astype("float32")
    y_val_p = y_scaler.transform(y_val).astype("float32")

    def build(widths):
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(8,))])
        for width in widths:
            model.add(tf.keras.layers.Dense(width, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        return model

    trials = []
    for widths in ((32, 16), (64, 32)):
        tf.keras.utils.set_random_seed(SEED)
        model = build(widths)
        history = model.fit(X_train_p, y_train_p, validation_data=(X_val_p, y_val_p),
                            epochs=300, batch_size=16, verbose=0,
                            callbacks=_callbacks(str(widths), 300, report, 25, 10))
        p_val = y_scaler.inverse_transform(model.predict(X_val_p, verbose=0)).ravel()
        trials.append((mean_absolute_error(y_val.ravel(), p_val), widths, model, history))
    best_mae, widths, model, history = min(trials, key=lambda item: item[0])
    del trials
    pred = y_scaler.inverse_transform(model.predict(X_test_p, verbose=0)).ravel()
    actual = y_test.ravel()
    baseline = np.full_like(actual, y_train.mean())
    return {
        "kind": "energy", "model": model, "X_test": X_test, "X_test_p": X_test_p,
        "y_test": y_test, "y_scaler": y_scaler, "configuration": widths, "history": _losses(history),
        "metrics": {
            "MAE": float(mean_absolute_error(actual, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "R²": float(r2_score(actual, pred)),
        },
        "baseline": float(mean_absolute_error(actual, baseline)),
        "sizes": (len(X_train), len(X_val), len(X_test)), "validation_score": float(best_mae),
        "actual": actual, "pred_test": pred,
    }


def train_forda(report: Report | None = None) -> dict:
    train_raw, test_raw = forda_data()
    X_dev = train_raw[:, 1:, None]
    y_dev = (train_raw[:, 0] == 1).astype("int32")
    X_test = test_raw[:, 1:, None]
    y_test = (test_raw[:, 0] == 1).astype("int32")
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=SEED)

    def build(filters):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(500, 1)),
            tf.keras.layers.Conv1D(filters[0], 5, padding="same"),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(filters[1], 5, padding="same"),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    trials = []
    for filters in ((32, 64), (64, 64)):
        tf.keras.utils.set_random_seed(SEED)
        model = build(filters)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=100, batch_size=64, verbose=0,
                            callbacks=_callbacks(str(filters), 100, report, 12, 5))
        p_val = model.predict(X_val, verbose=0).ravel()
        trials.append((f1_score(y_val, p_val >= 0.5), filters, model, history))
    best_f1, filters, model, history = max(trials, key=lambda item: item[0])
    del trials
    p_test = model.predict(X_test, verbose=0).ravel()
    pred = (p_test >= 0.5).astype("int32")
    return {
        "kind": "forda", "model": model, "X_test": X_test, "y_test": y_test,
        "configuration": filters, "history": _losses(history),
        "metrics": {"Exactitud": float(accuracy_score(y_test, pred)), "F1 (clase 1)": float(f1_score(y_test, pred))},
        "baseline": float(np.bincount(y_test).max() / len(y_test)),
        "confusion": confusion_matrix(y_test, pred),
        "sizes": (len(X_train), len(X_val), len(X_test)), "validation_score": float(best_f1),
    }


TRAINERS = {"bank": train_bank, "energy": train_energy, "forda": train_forda}


def predict_one(run: dict, index: int) -> dict:
    """Calcula una inferencia nueva con el modelo guardado en esta sesión."""
    if not 0 <= index < len(run["X_test"]):
        raise IndexError(f"El índice debe estar entre 0 y {len(run['X_test']) - 1}.")
    kind = run["kind"]
    if kind == "bank":
        probability = float(run["model"].predict(run["X_test_p"][index:index + 1], verbose=0).ravel()[0])
        return {"probability": probability, "prediction": int(probability >= run["threshold"]),
                "actual": int(run["y_test"][index]), "features": run["X_test"].iloc[[index]]}
    if kind == "energy":
        scaled = run["model"].predict(run["X_test_p"][index:index + 1], verbose=0)
        value = float(run["y_scaler"].inverse_transform(scaled)[0, 0])
        return {"prediction": value, "actual": float(run["y_test"][index, 0]),
                "features": pd.DataFrame([run["X_test"][index]], columns=[f"X{i}" for i in range(1, 9)])}
    probability = float(run["model"].predict(run["X_test"][index:index + 1], verbose=0).ravel()[0])
    return {"probability": probability, "prediction": int(probability >= 0.5),
            "actual": int(run["y_test"][index]), "signal": run["X_test"][index, :, 0]}
