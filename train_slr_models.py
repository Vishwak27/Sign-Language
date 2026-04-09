"""
train_slr_models.py — Three-Headed Bi-LSTM Training Pipeline
=============================================================
Strategy:
  1. Train a BASE model on ASL (largest dataset).
  2. Fine-tune ISL head  (two-handed SOV grammar adaptation).
  3. Fine-tune CSL head  (stroke-based character-mimicry adaptation).

Outputs:
  models/base_asl_model.keras
  models/isl_model.keras
  models/csl_model.keras
  models/base_asl_model.tflite
  models/isl_model.tflite
  models/csl_model.tflite
  reports/model_performance.md

Usage:
    python train_slr_models.py
    python train_slr_models.py --lang ASL        # Train only ASL base
    python train_slr_models.py --skip-base       # Skip base training, fine-tune only
"""

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF C++ warnings

import tensorflow as tf
import keras
from keras import layers, callbacks, regularizers

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LANDMARK_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    SEQUENCE_LENGTH, TOTAL_FEATURES,
    BATCH_SIZE, EPOCHS_BASE, EPOCHS_FINETUNE,
    LEARNING_RATE, FINETUNE_LR, DROPOUT_RATE,
    LSTM_UNITS, DENSE_UNITS,
)

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "training.log")),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_landmark_dataset(lang: str) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load all .npy landmark sequences for a given language.
    Returns (X, y_encoded, class_names).
    """
    lang_dir = Path(LANDMARK_DIR) / lang
    manifest_file = lang_dir / "manifest.json"

    if not lang_dir.exists():
        raise FileNotFoundError(
            f"Landmark directory not found: {lang_dir}\n"
            f"Run: python extract_landmarks.py --lang {lang} --demo"
        )

    X, y_raw = [], []
    classes_with_data = []

    for class_dir in sorted(lang_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        npy_files = list(class_dir.glob("*.npy"))
        if not npy_files:
            continue
        classes_with_data.append(class_dir.name)
        for npy_file in npy_files:
            seq = np.load(str(npy_file))
            if seq.shape == (SEQUENCE_LENGTH, TOTAL_FEATURES):
                X.append(seq)
                y_raw.append(class_dir.name)
            else:
                log.warning(f"Shape mismatch in {npy_file}: {seq.shape} — skipping.")

    if not X:
        raise ValueError(
            f"No valid landmark sequences found for {lang}. "
            f"Run extract_landmarks.py first."
        )

    X     = np.array(X, dtype=np.float32)
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    log.info(f"[{lang}] Loaded {len(X)} samples, {len(le.classes_)} classes.")
    return X, y_enc, list(le.classes_)


# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────
def build_bilstm_model(
    num_classes:    int,
    seq_len:        int    = SEQUENCE_LENGTH,
    feature_dim:    int    = TOTAL_FEATURES,
    lstm_units:     list   = LSTM_UNITS,
    dense_units:    list   = DENSE_UNITS,
    dropout_rate:   float  = DROPOUT_RATE,
    trainable_base: bool   = True,
    name:           str    = "SLR_BiLSTM",
) -> keras.Model:
    """
    Bi-directional LSTM classifier.

    Architecture:
        Input → Masking → BiLSTM × 2 → Dropout →
        Dense × 2 → Dropout → Softmax Output
    """
    inp = keras.Input(shape=(seq_len, feature_dim), name="landmarks")

    # Masking layer — handles zero-padded frames
    x = layers.Masking(mask_value=0.0)(inp)

    # Bi-LSTM stack
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        x = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=return_seq,
                kernel_regularizer=regularizers.l2(1e-4),
                recurrent_regularizer=regularizers.l2(1e-5),
                name=f"lstm_{i}",
            ),
            name=f"bilstm_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)

    x = layers.Dropout(dropout_rate, name="dropout_lstm")(x)

    # Dense classification head
    for j, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"dense_{j}",
        )(x)
        x = layers.Dropout(dropout_rate * 0.5, name=f"dropout_dense_{j}")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=name)
    return model


def freeze_base_layers(model: keras.Model, unfreeze_top_n: int = 2):
    """Freeze all layers except the top N for fine-tuning."""
    for layer in model.layers[:-unfreeze_top_n]:
        layer.trainable = False
    for layer in model.layers[-unfreeze_top_n:]:
        layer.trainable = True


# ─────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────
def get_callbacks(model_name: str, patience: int = 10) -> list:
    """Standard training callbacks."""
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    return [
        callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(LOGS_DIR, model_name),
            histogram_freq=1,
        ),
    ]


def train_model(
    model:       keras.Model,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    epochs:      int,
    lr:          float,
    model_name:  str,
) -> keras.callbacks.History:
    """Compile and train a model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log.info)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────────
# TRANSFER LEARNING — Fine-Tune Head
# ─────────────────────────────────────────────────────────────
def build_finetuned_model(
    base_model:    keras.Model,
    num_classes:   int,
    lang:          str,
) -> keras.Model:
    """
    Build a fine-tuned model by replacing the output layer of the base
    ASL model and freezing the shared BiLSTM backbone.
    """
    # Clone base model architecture
    new_model = keras.models.clone_model(base_model)
    new_model.set_weights(base_model.get_weights())

    # Freeze backbone (BiLSTM layers), keep dense head trainable
    freeze_base_layers(new_model, unfreeze_top_n=4)

    # Replace output layer
    inp = new_model.input
    x   = new_model.get_layer("dropout_dense_1").output   # Last shared dense
    out = layers.Dense(num_classes, activation="softmax",
                       name=f"output_{lang.lower()}")(x)

    finetuned = keras.Model(inputs=inp, outputs=out, name=f"SLR_{lang}_Head")
    return finetuned


# ─────────────────────────────────────────────────────────────
# EVALUATION & REPORTING
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, class_names, lang, history=None):
    """Run evaluation, generate plots and return metrics dict."""
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{lang} — Confusion Matrix  (Acc: {acc:.2%})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, f"{lang}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()

    # Training curves
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history["accuracy"],     label="Train")
        axes[0].plot(history.history["val_accuracy"], label="Val")
        axes[0].set_title(f"{lang} — Accuracy")
        axes[0].legend()

        axes[1].plot(history.history["loss"],     label="Train")
        axes[1].plot(history.history["val_loss"], label="Val")
        axes[1].set_title(f"{lang} — Loss")
        axes[1].legend()

        plt.tight_layout()
        curves_path = os.path.join(REPORTS_DIR, f"{lang}_training_curves.png")
        plt.savefig(curves_path, dpi=120)
        plt.close()

    log.info(f"[{lang}] Accuracy: {acc:.4f}")
    return {
        "accuracy":    round(acc, 4),
        "report":      report,
        "cm_path":     cm_path,
        "num_classes": len(class_names),
        "classes":     class_names,
    }


def export_tflite(model: keras.Model, name: str):
    """Export a Keras model to TFLite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path  = os.path.join(MODELS_DIR, f"{name}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    log.info(f"TFLite model saved -> {tflite_path}")
    return tflite_path


def generate_report(results: dict):
    """Write model_performance.md."""
    report_path = os.path.join(REPORTS_DIR, "model_performance.md")
    timestamp   = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Model Performance Report",
        f"> Generated: {timestamp}",
        "",
        "## Overview",
        "",
        "| Language | Accuracy | Classes | Architecture |",
        "|----------|----------|---------|--------------|",
    ]

    for lang, metrics in results.items():
        lines.append(
            f"| {lang} | {metrics['accuracy']:.2%} | "
            f"{metrics['num_classes']} | Bi-LSTM + Transfer Learning |"
        )

    lines += ["", "---", ""]

    for lang, metrics in results.items():
        acc = metrics["accuracy"]
        lines += [
            f"## {lang} Results",
            "",
            f"**Overall Accuracy:** `{acc:.2%}`",
            "",
            "### Per-Class Performance",
            "",
            "| Class | Precision | Recall | F1-Score | Support |",
            "|-------|-----------|--------|----------|---------|",
        ]
        rpt = metrics["report"]
        for cls in metrics["classes"]:
            if cls in rpt:
                r = rpt[cls]
                lines.append(
                    f"| `{cls}` | {r['precision']:.3f} | "
                    f"{r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |"
                )

        lines += [
            "",
            f"### Confusion Matrix",
            f"![{lang} Confusion Matrix]({lang}_confusion_matrix.png)",
            "",
            f"### Training Curves",
            f"![{lang} Training Curves]({lang}_training_curves.png)",
            "",
            "---",
            "",
        ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"Performance report saved -> {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SLR Training Pipeline")
    parser.add_argument("--lang",       default="all",
                        choices=["all", "ASL", "ISL", "CSL"])
    parser.add_argument("--skip-base",  action="store_true",
                        help="Skip ASL base training (load existing checkpoint)")
    args = parser.parse_args()

    results = {}

    # ── 1. ASL BASE MODEL ────────────────────────────────────
    if args.lang in ("all", "ASL") and not args.skip_base:
        log.info("=" * 60)
        log.info("PHASE 1 — Training ASL Base Model")
        log.info("=" * 60)

        X, y, classes = load_landmark_dataset("ASL")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

        log.info(f"ASL splits — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        base_model = build_bilstm_model(
            num_classes=len(classes),
            name="SLR_ASL_Base",
        )
        history = train_model(
            base_model, X_train, y_train, X_val, y_val,
            epochs=EPOCHS_BASE, lr=LEARNING_RATE,
            model_name="base_asl_model",
        )

        h5_path = os.path.join(MODELS_DIR, "base_asl_model.keras")
        base_model.save(h5_path)
        log.info(f"ASL base model saved -> {h5_path}")
        # export_tflite(base_model, "base_asl_model")

        # Save class names
        with open(os.path.join(MODELS_DIR, "asl_classes.json"), "w") as f:
            json.dump(classes, f, indent=2)

        results["ASL"] = evaluate_model(
            base_model, X_test, y_test, classes, "ASL", history
        )

    elif args.skip_base:
        # Load existing base model for transfer learning
        asl_path = os.path.join(MODELS_DIR, "base_asl_model.keras")
        if os.path.exists(asl_path):
            log.info(f"Loading existing ASL base model from {asl_path}")
            base_model = keras.models.load_model(asl_path)
        else:
            log.error("No existing ASL model found. Run without --skip-base first.")
            sys.exit(1)
    else:
        base_model = None

    # ── 2. ISL FINE-TUNED MODEL ───────────────────────────────
    if args.lang in ("all", "ISL"):
        log.info("=" * 60)
        log.info("PHASE 2 — Fine-tuning ISL Head (Two-Handed SOV Grammar)")
        log.info("=" * 60)

        try:
            X, y, classes = load_landmark_dataset("ISL")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
            )

            if base_model is not None:
                isl_model = build_finetuned_model(base_model, len(classes), "ISL")
            else:
                isl_model = build_bilstm_model(len(classes), name="SLR_ISL")

            history_isl = train_model(
                isl_model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS_FINETUNE, lr=FINETUNE_LR,
                model_name="isl_model",
            )

            isl_h5 = os.path.join(MODELS_DIR, "isl_model.keras")
            isl_model.save(isl_h5)
            log.info(f"ISL model saved -> {isl_h5}")
            # export_tflite(isl_model, "isl_model")

            with open(os.path.join(MODELS_DIR, "isl_classes.json"), "w") as f:
                json.dump(classes, f, indent=2)

            results["ISL"] = evaluate_model(
                isl_model, X_test, y_test, classes, "ISL", history_isl
            )
        except Exception as e:
            log.error(f"ISL training failed: {e}")

    # ── 3. CSL FINE-TUNED MODEL ───────────────────────────────
    if args.lang in ("all", "CSL"):
        log.info("=" * 60)
        log.info("PHASE 3 — Fine-tuning CSL Head (Stroke-Based Gestures)")
        log.info("=" * 60)

        try:
            X, y, classes = load_landmark_dataset("CSL")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
            )

            if base_model is not None:
                csl_model = build_finetuned_model(base_model, len(classes), "CSL")
            else:
                csl_model = build_bilstm_model(len(classes), name="SLR_CSL")

            history_csl = train_model(
                csl_model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS_FINETUNE, lr=FINETUNE_LR,
                model_name="csl_model",
            )

            csl_h5 = os.path.join(MODELS_DIR, "csl_model.keras")
            csl_model.save(csl_h5)
            log.info(f"CSL model saved -> {csl_h5}")
            # export_tflite(csl_model, "csl_model")

            with open(os.path.join(MODELS_DIR, "csl_classes.json"), "w") as f:
                json.dump(classes, f, indent=2)

            results["CSL"] = evaluate_model(
                csl_model, X_test, y_test, classes, "CSL", history_csl
            )
        except Exception as e:
            log.error(f"CSL training failed: {e}")

    # ── 4. REPORT ─────────────────────────────────────────────
    if results:
        report_path = generate_report(results)
        log.info("=" * 60)
        log.info(f"TRAINING COMPLETE — Report: {report_path}")
        log.info("=" * 60)
    else:
        log.warning("No models were trained. Check your setup and run extract_landmarks.py first.")


if __name__ == "__main__":
    main()
