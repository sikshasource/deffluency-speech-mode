#!/usr/bin/env python3
"""
FAST & ROBUST STUTTERING DYSFLUENCY DETECTION
=============================================

- Uses a pre-trained wav2vec2-like encoder as frozen feature extractor
- Adds a compact handcrafted prosodic/voice-quality feature vector
- Trains XGBoost + Linear SVM on pooled embeddings
- Caches embeddings to disk for fast retraining

Requirements:
    pip install torch transformers librosa scikit-learn xgboost numpy pandas tqdm joblib

Directory layout:
    segrigated_samples/
        fluent/
            sample1.wav
            sample2.wav
        disfluent/
            sample3.wav
            ...

This script will:
    1) Scan DATA_DIR for audio files
    2) Extract embeddings + handcrafted features (or load from cache)
    3) Train XGBoost and Linear SVM
    4) Evaluate on a held-out test set
    5) Save the best model + scaler + label encoder

NOTE: Actual accuracy depends on your data; 99% is not guaranteed but this
      is a strong, research-grade baseline.
"""

import os
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ========================= CONFIG =========================

DATA_DIR = "segrigated_samples"
CACHE_EMB_PATH = "embeddings_cache.npz"
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# You can swap this for other encoders, e.g. "microsoft/wavlm-base-plus"
ENCODER_MODEL_NAME = "facebook/wav2vec2-base"

TARGET_SR = 16000
MIN_SAMPLES = 1024
RANDOM_SEED = 42

# ========================= IMPORTS =========================

import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import LinearSVC

import xgboost as xgb
import joblib


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ========================= DATA DISCOVERY =========================


def list_audio_files(root: str) -> Tuple[List[str], List[str]]:
    """
    Traverse root/class_name/*.wav and return:
        paths: list of audio file paths
        labels: list of corresponding class names (folder names)
    """
    paths, labels = [], []
    for r, _, fs in os.walk(root):
        class_name = Path(r).name
        for f in fs:
            if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
                full = os.path.join(r, f)
                paths.append(full)
                labels.append(class_name)
    return paths, labels


# ========================= FEATURE EXTRACTOR =========================


class EmbeddingExtractor:
    """
    Frozen wav2vec2-like encoder + handcrafted features.
    """

    def __init__(self, model_name: str = ENCODER_MODEL_NAME, target_sr: int = TARGET_SR):
        print(f"Loading encoder: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.target_sr = target_sr
        self.embed_dim = self.model.config.hidden_size
        print(f"Encoder hidden size: {self.embed_dim}")

    def load_audio(self, path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(path, sr=self.target_sr, mono=True)
            if y.shape[0] < MIN_SAMPLES:
                # very short / invalid
                return np.array([], dtype=np.float32)
            return y.astype(np.float32)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return np.array([], dtype=np.float32)

    def extract_handcrafted(self, y: np.ndarray) -> np.ndarray:
        """
        40-dim prosodic/voice-quality features
        (MFCC statistics, ZCR, RMS, F0 mean/std).
        """
        out_dim = 40
        if y.size == 0:
            return np.zeros(out_dim, dtype=np.float32)

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=self.target_sr, n_mfcc=13)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=self.target_sr)

            f0_valid = f0[f0 > 0]
            f0_mean = float(np.mean(f0_valid)) if f0_valid.size > 0 else 0.0
            f0_std = float(np.std(f0_valid)) if f0_valid.size > 0 else 0.0

            feats = np.concatenate(
                [
                    np.mean(mfcc, axis=1),  # 13
                    np.std(mfcc, axis=1),   # 13
                    [np.mean(zcr), np.std(zcr)],  # 2
                    [np.mean(rms), np.std(rms)],  # 2
                    [f0_mean, f0_std],            # 2
                ]
            ).astype(np.float32)

            if feats.size < out_dim:
                feats = np.pad(feats, (0, out_dim - feats.size))
            elif feats.size > out_dim:
                feats = feats[:out_dim]

            return feats
        except Exception:
            return np.zeros(out_dim, dtype=np.float32)

    def extract_embedding(self, y: np.ndarray) -> np.ndarray:
        """
        Single utterance embedding: mean-pooled last hidden states.
        """
        if y.size == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)

        with torch.no_grad():
            inputs = self.processor(
                y,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(DEVICE)
            outputs = self.model(input_values)
            hidden_states = outputs.last_hidden_state  # (1, T, D)
            pooled = hidden_states.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

            # Ensure fixed length
            if pooled.shape[0] != self.embed_dim:
                out = np.zeros(self.embed_dim, dtype=np.float32)
                n = min(self.embed_dim, pooled.shape[0])
                out[:n] = pooled[:n]
                return out

            return pooled

    def extract_features_for_files(self, files: List[str]) -> np.ndarray:
        """
        Run full feature extraction for all files:
            [encoder embedding] + [handcrafted] -> (N, embed_dim + 40)
        """
        feats = []
        for path in tqdm(files, desc="Extracting features"):
            y = self.load_audio(path)
            emb = self.extract_embedding(y)
            hand = self.extract_handcrafted(y)
            full = np.concatenate([emb, hand]).astype(np.float32)
            feats.append(full)
        return np.vstack(feats)


# ========================= TRAINING & EVAL =========================


def train_and_evaluate(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder):
    """
    Train XGBoost + Linear SVM; select best by macro F1, evaluate on test set,
    and save best model + scaler + encoder.
    """
    # 70/15/15 split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15
        stratify=y_train_val,
        random_state=RANDOM_SEED,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = {c: w for c, w in zip(classes, class_weights)}
    sample_weight = np.array([weight_map[yi] for yi in y_train], dtype=np.float32)

    # ---- XGBoost ----
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",  # IMPORTANT: here (constructor), not in fit()
    )
    xgb_model.fit(X_train_s, y_train, sample_weight=sample_weight)

    val_probs_xgb = xgb_model.predict_proba(X_val_s)
    val_pred_xgb = np.argmax(val_probs_xgb, axis=1)
    acc_xgb = accuracy_score(y_val, val_pred_xgb)
    prec_xgb, rec_xgb, f1_xgb, _ = precision_recall_fscore_support(
        y_val, val_pred_xgb, average="macro", zero_division=0
    )
    print("\nXGBoost validation:")
    print(
        f"  Acc: {acc_xgb:.4f} | Prec: {prec_xgb:.4f} | "
        f"Rec: {rec_xgb:.4f} | F1: {f1_xgb:.4f}"
    )

    # ---- Linear SVM ----
    svm_model = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)
    svm_model.fit(X_train_s, y_train)
    val_pred_svm = svm_model.predict(X_val_s)

    acc_svm = accuracy_score(y_val, val_pred_svm)
    prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(
        y_val, val_pred_svm, average="macro", zero_division=0
    )
    print("\nLinear SVM validation:")
    print(
        f"  Acc: {acc_svm:.4f} | Prec: {prec_svm:.4f} | "
        f"Rec: {rec_svm:.4f} | F1: {f1_svm:.4f}"
    )

    # ---- Select best by macro F1 ----
    if f1_xgb >= f1_svm:
        best_name = "xgboost"
        best_model = xgb_model
    else:
        best_name = "svm"
        best_model = svm_model

    print(f"\nSelected best model: {best_name}")

    # ---- Test evaluation ----
    if best_name == "xgboost":
        test_probs = best_model.predict_proba(X_test_s)
        test_pred = np.argmax(test_probs, axis=1)
    else:
        test_pred = best_model.predict(X_test_s)

    acc_test = accuracy_score(y_test, test_pred)
    prec_test, rec_test, f1_test, _ = precision_recall_fscore_support(
        y_test, test_pred, average="macro", zero_division=0
    )
    print("\n=== Test Performance ===")
    print(
        f"Acc: {acc_test:.4f} | Prec: {prec_test:.4f} | "
        f"Rec: {rec_test:.4f} | F1: {f1_test:.4f}"
    )

    # ---- Save artifacts ----
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"best_model_{best_name}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    print(
        f"\nSaved best model ({best_name}), scaler, and label encoder to '{MODEL_DIR}'"
    )


# ========================= MAIN =========================


def main():
    # 1) Collect files and labels
    files, labels = list_audio_files(DATA_DIR)
    if not files:
        print(f"No audio files found under {DATA_DIR}/")
        return

    print(f"Found {len(files)} files across {len(set(labels))} classes: {set(labels)}")

    # 2) Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # 3) Load cached embeddings if present
    X = None
    if os.path.exists(CACHE_EMB_PATH):
        try:
            print(f"Loading cached embeddings from {CACHE_EMB_PATH}")
            data = np.load(CACHE_EMB_PATH)
            X_cached = data["X"]
            y_cached = data["y"]
            if X_cached.shape[0] == len(files) and y_cached.shape[0] == len(files):
                X = X_cached
                y = y_cached
            else:
                print("Cache size mismatch with current dataset; recomputing embeddings.")
        except Exception as e:
            print(f"Failed to load cache ({e}); recomputing embeddings.")

    # 4) Compute embeddings if no valid cache
    if X is None:
        extractor = EmbeddingExtractor()
        X = extractor.extract_features_for_files(files)
        np.savez(CACHE_EMB_PATH, X=X, y=y)
        print(f"Saved embeddings cache to {CACHE_EMB_PATH}")

    # 5) Train & evaluate models
    if X.shape[0] < 10:
        print("Not enough samples to train a reliable model.")
        return

    train_and_evaluate(X, y, le)


if __name__ == "__main__":
    main()