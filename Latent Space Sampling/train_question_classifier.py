#!/usr/bin/env python3
"""
Train an L2-normalized question-vs-statement classifier.

This script prepares and trains the question-form classifier used for filtering
generated benchmark candidates. It preserves the original notebook logic:

    1. Load a question-vs-statement CSV dataset.
    2. Encode text with sentence-transformers/gtr-t5-base.
    3. Mean-pool encoder hidden states.
    4. L2-normalize embeddings.
    5. Split into train / validation / test sets.
    6. Standardize embeddings using train-set mean and std.
    7. Train a residual MLP binary classifier.
    8. Select the decision threshold on validation F1.
    9. Evaluate on the held-out test set.
    10. Save the classifier weights and normalization / threshold statistics.

Required packages:
    pip install torch transformers scikit-learn numpy pandas tqdm

Example:
    python train_question_classifier.py \
        --csv-path ./data/questions_vs_statements_v1.0.csv \
        --output-dir ./artifacts/question_detector_gtr_t5_base_resnet_L2 \
        --text-column doc \
        --label-column target \
        --embedding-cache ./artifacts/qvstmt_gtr_t5_base_embeddings_L2.npy \
        --seed 2026 \
        --split-seed 2025 \
        --batch-size 512 \
        --embedding-batch-size 256 \
        --max-epochs 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_ENCODER_MODEL = "sentence-transformers/gtr-t5-base"
DEFAULT_MODEL_FILENAME = "resnet_qvstmt_gtr_t5_base_L2.pt"
DEFAULT_STATS_FILENAME = "standardize_mu_sg_and_threshold_L2.npz"
DEFAULT_METRICS_FILENAME = "question_classifier_metrics_L2.json"


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for classifier training."""

    csv_path: Path
    output_dir: Path
    text_column: str
    label_column: str
    encoder_model: str
    embedding_cache: Optional[Path]
    force_reencode: bool
    force_retrain: bool

    seed: int
    split_seed: int
    val_ratio: float
    test_ratio: float

    max_length: int
    embedding_batch_size: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    patience: int

    width: int
    num_blocks: int
    proj_dropout: float
    block_dropout: float

    threshold_min: float
    threshold_max: float
    threshold_steps: int

    device: str
    num_workers: int


class ResidualBlock(nn.Module):
    """Residual fully connected block used by the question classifier."""

    def __init__(self, dim: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.act(out + x)


class ResNetBinaryClassifier(nn.Module):
    """
    Residual MLP binary classifier.

    This preserves the original ResNet-style classifier architecture:
        Linear -> BatchNorm -> ReLU -> Dropout
        ResidualBlock x num_blocks
        Linear head to one logit
    """

    def __init__(
        self,
        in_dim: int = 768,
        width: int = 512,
        num_blocks: int = 4,
        proj_dropout: float = 0.30,
        block_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(proj_dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, dropout=block_dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert binary logits to probabilities."""
    return 1.0 / (1.0 + np.exp(-logits))


def validate_ratios(val_ratio: float, test_ratio: float) -> None:
    """Validate train / validation / test split ratios."""
    if val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("--val-ratio and --test-ratio must both be positive.")

    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            "--val-ratio + --test-ratio must be less than 1.0 "
            "so that the train split is non-empty."
        )


def load_question_statement_csv(
    csv_path: Path,
    *,
    text_column: str,
    label_column: str,
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Load and clean the question-vs-statement dataset.

    Expected default columns follow the original notebook:
        doc    -> text
        target -> label
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in (text_column, label_column) if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    try:
        df["label"] = df["label"].astype(int)
    except ValueError as exc:
        raise ValueError(
            f"Label column '{label_column}' must contain integer binary labels."
        ) from exc

    unique_labels = sorted(df["label"].unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            f"Expected binary labels [0, 1], got {unique_labels}. "
            "Please remap labels before training."
        )

    class_counts = df["label"].value_counts().to_dict()
    if min(class_counts.values()) < 3:
        raise ValueError(
            f"Each class must contain at least 3 examples for stratified splitting. "
            f"Observed counts: {class_counts}"
        )

    texts = df["text"].tolist()
    labels = df["label"].values.astype(np.int64)

    return texts, labels, df


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token representations using the attention mask."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts_l2(
    texts: Sequence[str],
    *,
    encoder_model: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """
    Encode texts with a transformer encoder and return L2-normalized embeddings.

    The original notebook used sentence-transformers/gtr-t5-base, took the encoder
    outputs, mean-pooled them, and applied L2 normalization.
    """
    if not texts:
        raise ValueError("No texts provided for encoding.")

    print(f"[INFO] Loading encoder: {encoder_model}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    model = AutoModel.from_pretrained(encoder_model).to(device)

    encoder = model.encoder if hasattr(model, "encoder") else model
    encoder.eval()

    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Encoding L2 embeddings ({encoder_model})",
        ):
            batch = list(texts[start : start + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def load_or_create_embeddings(
    texts: Sequence[str],
    *,
    config: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    """Load cached L2 embeddings or encode them from scratch."""
    if config.embedding_cache and config.embedding_cache.exists() and not config.force_reencode:
        print(f"[INFO] Loading cached L2 embeddings: {config.embedding_cache}")
        embeddings = np.load(config.embedding_cache).astype(np.float32)
    else:
        print("[INFO] Encoding L2 embeddings from raw text.")
        embeddings = encode_texts_l2(
            texts,
            encoder_model=config.encoder_model,
            device=device,
            batch_size=config.embedding_batch_size,
            max_length=config.max_length,
        )

        if config.embedding_cache:
            config.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(config.embedding_cache, embeddings)
            print(f"[SAVE] Embedding cache: {config.embedding_cache}")

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Embedding count does not match text count: "
            f"{embeddings.shape[0]} embeddings vs {len(texts)} texts."
        )

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[INFO] Embedding shape: {embeddings.shape}")
    print(f"[INFO] L2 norm mean/std: {norms.mean():.6f} / {norms.std():.6f}")

    return embeddings


def standardize_train_val_test(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize embeddings using train-set mean and std.

    Returns:
        x_train_norm, x_val_norm, x_test_norm, mu, sg
    """
    mu = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    sg = x_train.std(axis=0, keepdims=True).astype(np.float32)
    sg[sg < 1e-6] = 1.0

    def normalize(x: np.ndarray) -> np.ndarray:
        return ((x - mu) / sg).astype(np.float32)

    return normalize(x_train), normalize(x_val), normalize(x_test), mu, sg


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> tud.DataLoader:
    """Create a PyTorch DataLoader for dense embeddings."""
    dataset = tud.TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def collect_logits(
    model: nn.Module,
    loader: tud.DataLoader,
    *,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect logits and labels from a DataLoader."""
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device)).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(yb.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels).astype(int)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC, returning NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC / average precision, returning NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def choose_threshold_by_validation_f1(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Tuple[float, float]:
    """Select a probability threshold that maximizes validation F1."""
    grid = np.linspace(threshold_min, threshold_max, threshold_steps)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in grid:
        pred = (val_probs >= threshold).astype(int)
        score = f1_score(y_val, pred, zero_division=0)

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_binary_classifier(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, object]:
    """Evaluate binary classifier predictions at a fixed threshold."""
    pred = (probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )

    metrics: Dict[str, object] = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": safe_roc_auc(y_true, probs),
        "pr_auc": safe_average_precision(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }

    return metrics


def train_question_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[ResNetBinaryClassifier, np.ndarray, np.ndarray, float, Dict[str, object]]:
    """Train, tune, evaluate, and return the L2 question classifier."""
    x_train, x_rest, y_train, y_rest = train_test_split(
        embeddings,
        labels,
        test_size=(config.val_ratio + config.test_ratio),
        random_state=config.split_seed,
        stratify=labels,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_rest,
        y_rest,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        random_state=config.split_seed,
        stratify=y_rest,
    )

    x_train_n, x_val_n, x_test_n, mu, sg = standardize_train_val_test(
        x_train,
        x_val,
        x_test,
    )

    model = ResNetBinaryClassifier(
        in_dim=embeddings.shape[1],
        width=config.width,
        num_blocks=config.num_blocks,
        proj_dropout=config.proj_dropout,
        block_dropout=config.block_dropout,
    ).to(device)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    pos_weight = torch.tensor(
        [max(1.0, n_neg / max(1, n_pos))],
        dtype=torch.float32,
        device=device,
    )

    print(f"[INFO] Train split: {len(y_train)} examples")
    print(f"[INFO] Validation split: {len(y_val)} examples")
    print(f"[INFO] Test split: {len(y_test)} examples")
    print(f"[INFO] pos_weight={pos_weight.item():.4f} (pos={n_pos}, neg={n_neg})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
    )

    train_loader = make_loader(
        x_train_n,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = make_loader(
        x_val_n,
        y_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = make_loader(
        x_test_n,
        y_test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    best_val_f1_at_05 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)

        scheduler.step()

        val_logits, val_labels = collect_logits(model, val_loader, device=device)
        val_probs = logits_to_probs(val_logits)
        val_pred_05 = (val_probs >= 0.5).astype(int)

        val_acc = accuracy_score(val_labels, val_pred_05)
        val_f1 = f1_score(val_labels, val_pred_05, zero_division=0)
        val_auc = safe_roc_auc(val_labels, val_probs)

        mean_loss = total_loss / max(len(x_train_n), 1)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={mean_loss:.4f} "
            f"val_acc@0.5={val_acc:.4f} "
            f"val_F1@0.5={val_f1:.4f} "
            f"val_ROC-AUC={val_auc:.4f}"
        )

        if val_f1 > best_val_f1_at_05 + 1e-4:
            best_val_f1_at_05 = float(val_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= config.patience:
                print(
                    f"[early-stop] best_epoch={best_epoch}, "
                    f"best_val_F1@0.5={best_val_f1_at_05:.4f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_labels = collect_logits(model, val_loader, device=device)
    val_probs = logits_to_probs(val_logits)

    best_threshold, best_val_f1 = choose_threshold_by_validation_f1(
        val_labels,
        val_probs,
        threshold_min=config.threshold_min,
        threshold_max=config.threshold_max,
        threshold_steps=config.threshold_steps,
    )

    print(f"[THRESH] best_th={best_threshold:.3f} best_val_F1={best_val_f1:.4f}")

    test_logits, test_labels = collect_logits(model, test_loader, device=device)
    test_probs = logits_to_probs(test_logits)
    test_metrics = evaluate_binary_classifier(
        test_labels,
        test_probs,
        threshold=best_threshold,
    )

    print("\n[TEST] New L2 question classifier with validation-selected threshold")
    print(
        f"Acc={test_metrics['accuracy']:.4f} "
        f"F1={test_metrics['f1']:.4f} "
        f"Prec={test_metrics['precision']:.4f} "
        f"Rec={test_metrics['recall']:.4f} "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"PR-AUC={test_metrics['pr_auc']:.4f}"
    )
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(np.array(test_metrics["confusion_matrix"]))

    all_metrics: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_val_f1_at_threshold_0_5": best_val_f1_at_05,
        "best_threshold": best_threshold,
        "best_val_f1": best_val_f1,
        "test": test_metrics,
        "train_size": int(len(y_train)),
        "validation_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "embedding_dim": int(embeddings.shape[1]),
        "positive_train_examples": n_pos,
        "negative_train_examples": n_neg,
        "pos_weight": float(pos_weight.item()),
    }

    return model.eval(), mu, sg, best_threshold, all_metrics


def save_artifacts(
    *,
    model: ResNetBinaryClassifier,
    mu: np.ndarray,
    sg: np.ndarray,
    threshold: float,
    metrics: Dict[str, object],
    config: TrainConfig,
) -> None:
    """Save classifier weights, standardization stats, threshold, and metrics."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = config.output_dir / DEFAULT_MODEL_FILENAME
    stats_path = config.output_dir / DEFAULT_STATS_FILENAME
    metrics_path = config.output_dir / DEFAULT_METRICS_FILENAME
    config_path = config.output_dir / "train_config.json"

    torch.save(model.state_dict(), model_path)

    test_metrics = metrics.get("test", {})
    if not isinstance(test_metrics, dict):
        test_metrics = {}

    np.savez(
        stats_path,
        mu=mu.astype(np.float32),
        sg=sg.astype(np.float32),
        best_th=np.array([threshold], dtype=np.float32),
        best_val_f1=np.array([metrics.get("best_val_f1", np.nan)], dtype=np.float32),
        test_acc=np.array([test_metrics.get("accuracy", np.nan)], dtype=np.float32),
        test_f1=np.array([test_metrics.get("f1", np.nan)], dtype=np.float32),
        test_precision=np.array([test_metrics.get("precision", np.nan)], dtype=np.float32),
        test_recall=np.array([test_metrics.get("recall", np.nan)], dtype=np.float32),
        test_roc_auc=np.array([test_metrics.get("roc_auc", np.nan)], dtype=np.float32),
        test_pr_auc=np.array([test_metrics.get("pr_auc", np.nan)], dtype=np.float32),
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    config_dict = asdict(config)
    config_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config_dict.items()
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] Classifier model: {model_path}")
    print(f"[SAVE] Standardization stats and threshold: {stats_path}")
    print(f"[SAVE] Metrics: {metrics_path}")
    print(f"[SAVE] Training config: {config_path}")


def load_classifier(
    *,
    model_path: Path,
    stats_path: Path,
    in_dim: int,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[ResNetBinaryClassifier, np.ndarray, np.ndarray, float]:
    """Load a saved question classifier and its normalization statistics."""
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")

    model = ResNetBinaryClassifier(
        in_dim=in_dim,
        width=config.width,
        num_blocks=config.num_blocks,
        proj_dropout=config.proj_dropout,
        block_dropout=config.block_dropout,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    stats = np.load(stats_path)
    mu = stats["mu"].astype(np.float32)
    sg = stats["sg"].astype(np.float32)

    if "best_th" in stats.files:
        threshold = float(stats["best_th"][0])
    elif "threshold" in stats.files:
        threshold = float(stats["threshold"][0])
    else:
        raise KeyError(f"No threshold found in stats file: {stats_path}")

    return model, mu, sg, threshold


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an L2-normalized question-vs-statement classifier."
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to the question-vs-statement CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where classifier artifacts will be saved.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="doc",
        help="Name of the text column in the CSV. Default: doc.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="target",
        help="Name of the binary label column in the CSV. Default: target.",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default=DEFAULT_ENCODER_MODEL,
        help=f"Transformer encoder used for embeddings. Default: {DEFAULT_ENCODER_MODEL}.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Optional .npy path for cached L2 embeddings.",
    )
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Recompute embeddings even if --embedding-cache already exists.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if saved model and stats already exist.",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=6)

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--proj-dropout", type=float, default=0.30)
    parser.add_argument("--block-dropout", type=float, default=0.20)

    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=181)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Default: 0.",
    )

    args = parser.parse_args(argv)

    validate_ratios(args.val_ratio, args.test_ratio)

    if args.threshold_steps < 2:
        raise ValueError("--threshold-steps must be at least 2.")

    if not 0.0 <= args.threshold_min < args.threshold_max <= 1.0:
        raise ValueError(
            "Expected 0.0 <= --threshold-min < --threshold-max <= 1.0."
        )

    return TrainConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        encoder_model=args.encoder_model,
        embedding_cache=args.embedding_cache,
        force_reencode=args.force_reencode,
        force_retrain=args.force_retrain,
        seed=args.seed,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_length=args.max_length,
        embedding_batch_size=args.embedding_batch_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        width=args.width,
        num_blocks=args.num_blocks,
        proj_dropout=args.proj_dropout,
        block_dropout=args.block_dropout,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        device=args.device,
        num_workers=args.num_workers,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run classifier preparation, training, evaluation, and saving."""
    config = parse_args(argv)
    set_seed(config.seed)
    device = resolve_device(config.device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] CSV path: {config.csv_path}")
    print(f"[INFO] Output directory: {config.output_dir}")

    texts, labels, df = load_question_statement_csv(
        config.csv_path,
        text_column=config.text_column,
        label_column=config.label_column,
    )

    print(f"[INFO] Dataset size: {len(df)}")
    print("[INFO] Label distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    embeddings = load_or_create_embeddings(
        texts,
        config=config,
        device=device,
    )

    model_path = config.output_dir / DEFAULT_MODEL_FILENAME
    stats_path = config.output_dir / DEFAULT_STATS_FILENAME

    if model_path.exists() and stats_path.exists() and not config.force_retrain:
        print("[INFO] Existing classifier artifacts found.")
        print("[INFO] Use --force-retrain to train a new classifier.")
        model, mu, sg, threshold = load_classifier(
            model_path=model_path,
            stats_path=stats_path,
            in_dim=embeddings.shape[1],
            config=config,
            device=device,
        )
        print(f"[INFO] Loaded classifier threshold: {threshold:.4f}")
        print(f"[INFO] Loaded mu shape: {mu.shape}")
        print(f"[INFO] Loaded sg shape: {sg.shape}")
        return 0

    print("[INFO] Training new L2 question classifier.")
    model, mu, sg, threshold, metrics = train_question_classifier(
        embeddings,
        labels,
        config=config,
        device=device,
    )

    save_artifacts(
        model=model,
        mu=mu,
        sg=sg,
        threshold=threshold,
        metrics=metrics,
        config=config,
    )

    print(f"[INFO] Final validation-selected threshold: {threshold:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user.", file=sys.stderr)
        raise SystemExit(130)