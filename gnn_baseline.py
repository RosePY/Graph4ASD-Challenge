import os
import time
import random
import platform
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Config
# ============================================================

SEED = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADJ_TRAIN_PATH = "data/public/adj_train.npy"
ADJ_TEST_PATH = "data/public/adj_test.npy"
LABEL_PATH = "data/public/train_label.csv"
SAMPLE_SUB_PATH = "data/public/sample_submission.csv"

PREDICTION_PATH = "predictions.csv"
LOG_PATH = "run_log.txt"
LOSS_CSV_PATH = "loss_history.csv"

NUM_EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
DROPOUT = 0.2
EARLY_STOPPING_PATIENCE = 15


# ============================================================
# 2. Seed
# ============================================================

def set_seed(seed: int = 25) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Data
# ============================================================

def preprocess_connectivity(a: np.ndarray) -> np.ndarray:
    """
    Clean connectivity matrix for GCN usage:
    - symmetrize
    - clip negatives to 0
    """
    a = np.asarray(a, dtype=np.float32)
    a = 0.5 * (a + a.T)
    a = np.maximum(a, 0.0)
    return a


def normalize_adjacency(a: np.ndarray) -> np.ndarray:
    """
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = np.sum(a, axis=1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-8), -0.5)
    d_inv_sqrt = np.diag(deg_inv_sqrt)
    return (d_inv_sqrt @ a @ d_inv_sqrt).astype(np.float32)


def load_data():
    adj_train_raw = np.load(ADJ_TRAIN_PATH).astype(np.float32)
    adj_test_raw = np.load(ADJ_TEST_PATH).astype(np.float32)
    y = pd.read_csv(LABEL_PATH)["label"].values.astype(np.int64)

    # Use adjacency both as graph structure and as node features
    adj_train_clean = np.stack([preprocess_connectivity(a) for a in adj_train_raw], axis=0)
    adj_test_clean = np.stack([preprocess_connectivity(a) for a in adj_test_raw], axis=0)

    x_train = adj_train_clean.copy()
    x_test = adj_test_clean.copy()

    adj_train = np.stack([normalize_adjacency(a) for a in adj_train_clean], axis=0)
    adj_test = np.stack([normalize_adjacency(a) for a in adj_test_clean], axis=0)

    return adj_train, x_train, y, adj_test, x_test


# ============================================================
# 4. Model
# ============================================================

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.bmm(adj, x)
        x = self.linear(x)
        return x


class UpgradedGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gcn2(x, adj)
        x = F.relu(x)

        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        g = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(g)


# ============================================================
# 5. Training
# ============================================================

@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list
    val_losses: list
    val_f1s: list
    best_val_f1: float
    training_seconds: float


def evaluate(model, x, adj, y):
    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average="macro")
    return loss.item(), f1


def train_model(model, x_tr, adj_tr, y_tr, x_val, adj_val, y_val):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = -1.0
    best_state = None
    patience = 0

    train_losses = []
    val_losses = []
    val_f1s = []

    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_tr, adj_tr)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        val_loss, val_f1 = evaluate(model, x_val, adj_val, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    duration = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        val_f1s=val_f1s,
        best_val_f1=best_f1,
        training_seconds=duration,
    )


def save_loss_history(train_losses, val_losses, val_f1s, path):
    pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_macro_f1": val_f1s,
    }).to_csv(path, index=False)


def write_log(best_val_f1: float, training_seconds: float):
    lines = [
        "Graph4ASD Challenge - Compliance Log",
        "====================================",
        f"Seed: {SEED}",
        "Model: Upgraded 2-layer GCN",
        "Features: adjacency matrix used as node features",
        "Pooling: mean + max pooling",
        "Search strategy: manual",
        "Number of runs used: 1",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Hidden dimension: {HIDDEN_DIM}",
        f"Dropout: {DROPOUT}",
        f"Epoch budget: {NUM_EPOCHS}",
        f"Early stopping patience: {EARLY_STOPPING_PATIENCE}",
        f"Best validation Macro-F1: {best_val_f1:.6f}",
        f"Training time (seconds): {training_seconds:.2f}",
        f"Loss history file: {LOSS_CSV_PATH}",
        "",
        "Hardware",
        "--------",
        f"Platform: {platform.platform()}",
        f"Processor: {platform.processor()}",
        f"PyTorch version: {torch.__version__}",
        f"Device used: {DEVICE}",
    ]
    if torch.cuda.is_available():
        lines.append(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. Main
# ============================================================

def main():
    set_seed(SEED)

    adj_train, x_train, y, adj_test, x_test = load_data()

    idx = np.arange(len(y))
    tr_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    x_tr = torch.tensor(x_train[tr_idx], dtype=torch.float32, device=DEVICE)
    adj_tr = torch.tensor(adj_train[tr_idx], dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.long, device=DEVICE)

    x_val = torch.tensor(x_train[val_idx], dtype=torch.float32, device=DEVICE)
    adj_val = torch.tensor(adj_train[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = torch.tensor(y[val_idx], dtype=torch.long, device=DEVICE)

    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    adj_test_t = torch.tensor(adj_test, dtype=torch.float32, device=DEVICE)

    model = UpgradedGCN(
        in_dim=x_train.shape[-1],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    result = train_model(model, x_tr, adj_tr, y_tr, x_val, adj_val, y_val)

    result.model.eval()
    with torch.no_grad():
        logits = result.model(x_test_t, adj_test_t)
        pred = logits.argmax(dim=1).cpu().numpy().astype(int)

    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    if len(sample_sub) != len(pred):
        raise ValueError(
            f"Prediction length mismatch: got {len(pred)} predictions, "
            f"but sample submission expects {len(sample_sub)} rows."
        )

    submission = pd.DataFrame({
        "id": sample_sub["id"].values,
        "y_pred": pred,
    })
    submission.to_csv(PREDICTION_PATH, index=False)

    save_loss_history(result.train_losses, result.val_losses, result.val_f1s, LOSS_CSV_PATH)
    write_log(result.best_val_f1, result.training_seconds)

    print("\nDONE")
    print(f"Best validation Macro-F1: {result.best_val_f1:.6f}")
    print(f"Training time: {result.training_seconds:.2f} seconds")
    print(f"Saved predictions to: {PREDICTION_PATH}")
    print(f"Saved loss history to: {LOSS_CSV_PATH}")
    print(f"Saved run log to: {LOG_PATH}")


if __name__ == "__main__":
    main()
