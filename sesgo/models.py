"""Clasificadores de polaridad (3 clases: NEG/NEU/POS) con interfaz comun.

Lineas base EJECUTABLES sin red ni GPU: mayoritaria y TF-IDF + regresion logistica.
Backends de transformers (BETO, RoBERTuito/pysentimiento) estan implementados con
importacion perezosa; requieren `pip install -r requirements-beto.txt`, acceso a
Hugging Face Hub y, para el ajuste fino, GPU. NO se ejecutaron en el entorno de
desarrollo de este avance (sin acceso a Hugging Face).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .config import CLASSES, config_hash
from .db import now
from .splits import load_role


class Classifier(Protocol):
    def fit(self, texts: list[str], y: list[str]) -> "Classifier": ...
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...  # columnas = CLASSES


class MajorityBaseline:
    def fit(self, texts, y):
        self.label_ = pd.Series(list(y)).value_counts().idxmax()
        return self

    def predict_proba(self, texts):
        P = np.zeros((len(texts), len(CLASSES)))
        P[:, CLASSES.index(self.label_)] = 1.0
        return P


class TfidfLogReg:
    def __init__(self, C: float = 1.0, seed: int = 0):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
            ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)),
        ])

    def fit(self, texts, y):
        self.pipe.fit(texts, y)
        return self

    def predict_proba(self, texts):
        P = self.pipe.predict_proba(texts)
        cls = list(self.pipe.classes_)
        out = np.zeros((len(texts), len(CLASSES)))
        for j, c in enumerate(cls):
            out[:, CLASSES.index(c)] = P[:, j]
        return out


class BetoClassifier:  # pragma: no cover - requiere transformers/torch/GPU y Hugging Face Hub
    """BETO (bert-base-spanish-wwm-cased) + cabeza lineal de clasificacion (3 clases)."""

    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-cased", epochs=3, lr=2e-5,
                 batch_size=16, max_length=128, seed=0):
        self.model_name, self.epochs, self.lr = model_name, epochs, lr
        self.batch_size, self.max_length, self.seed = batch_size, max_length, seed

    def fit(self, texts, y):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(self.seed)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(CLASSES),
            id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)}).to(self.dev)
        yi = torch.tensor([CLASSES.index(v) for v in y])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            perm = np.random.RandomState(self.seed).permutation(len(texts))
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                enc = self.tok([texts[j] for j in idx], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.dev)
                loss = self.model(**enc, labels=yi[idx].to(self.dev)).loss
                loss.backward(); opt.step(); opt.zero_grad()
        return self

    def predict_proba(self, texts):
        import torch
        self.model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = self.tok(texts[i:i + 64], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.dev)
                out.append(torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return np.vstack(out)


class RobertuitoSentiment:  # pragma: no cover - requiere pysentimiento y Hugging Face Hub
    """Linea base sin ajuste fino: pysentimiento/robertuito-sentiment-analysis (POS/NEG/NEU)."""

    def fit(self, texts, y):
        from pysentimiento import create_analyzer
        self.an = create_analyzer(task="sentiment", lang="es")
        return self

    def predict_proba(self, texts):
        out = np.zeros((len(texts), len(CLASSES)))
        for i, r in enumerate(self.an.predict(list(texts))):
            for k, v in r.probas.items():
                out[i, CLASSES.index(k)] = v
        return out


REGISTRY = {"majority": MajorityBaseline, "tfidf_lr": TfidfLogReg,
            "beto": BetoClassifier, "robertuito": RobertuitoSentiment}


def run_experiment(conn: sqlite3.Connection, name: str, backend: str, split_version: str,
                   cfg: dict, params: dict | None = None) -> int:
    params = params or {}
    clf = REGISTRY[backend](**params)
    cur = conn.execute("INSERT INTO model_run(name, backend, split_version, params_json, config_hash, started_at) "
                       "VALUES (?,?,?,?,?,?)", (name, backend, split_version, json.dumps(params), config_hash(cfg), now()))
    run_id = cur.lastrowid
    train, dev = load_role(conn, split_version, "train"), load_role(conn, split_version, "dev")
    test, audit = load_role(conn, split_version, "test"), load_role(conn, split_version, "audit")
    clf.fit(train["text_norm"].tolist(), train["y_class"].tolist())
    metrics = {"n_train": len(train)}
    rows = []
    for role, part in (("dev", dev), ("test", test), ("audit", audit)):
        if part.empty:
            continue
        P = clf.predict_proba(part["text_norm"].tolist())
        pred = [CLASSES[i] for i in P.argmax(1)]
        if role != "audit":  # el conjunto de auditoria solo se evalua en fairness.run_audit
            metrics[f"macro_f1_{role}"] = float(f1_score(part["y_class"], pred, labels=list(CLASSES),
                                                         average="macro", zero_division=0))
        if role in ("test", "audit"):
            rows += [(run_id, rid, p, json.dumps(dict(zip(CLASSES, map(float, pr)))))
                     for rid, p, pr in zip(part["id"], pred, P)]
    conn.executemany("INSERT INTO prediction(run_id, review_id, y_pred, proba_json) VALUES (?,?,?,?)", rows)
    conn.execute("UPDATE model_run SET metrics_json=?, finished_at=? WHERE id=?",
                 (json.dumps(metrics), now(), run_id))
    conn.commit()
    return run_id
