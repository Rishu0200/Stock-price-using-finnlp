# ──────────────────────────────────────────────────────────────
#  src/model_trainer/lstm_trainer.py
#  Bidirectional-capable LSTM for stock direction classification.
#  Uses sliding-window sequences (SEQ_LEN trading days → predict day+1).
#  Implements Phase 6 Sections 5 notebook logic as a trainer class.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config.config import settings
from config.constants import LSTM_MODEL_FILE
from src.model_trainer.base_trainer import BaseTrainer
from utils.logging_utils import get_logger

log = get_logger(__name__)


# ── Dataset ───────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """
    Sliding-window dataset for LSTM.
    __getitem__(i) returns:
      x : FloatTensor [SEQ_LEN, n_features]  — last SEQ_LEN days of features
      y : FloatTensor scalar                  — direction of day i + SEQ_LEN
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return x, y


# ── Model architecture ────────────────────────────────────────

class StockLSTM(nn.Module):
    """
    LSTM for binary classification of next-day stock direction.

    Architecture:
      Input  [batch, seq_len, n_features]
        → LSTM (num_layers, optional bidirectional)
        → Final hidden state [batch, hidden_size (* 2 if bidir)]
        → Dropout
        → Linear → ReLU → Linear
      Output [batch, 1]  (raw logit — apply sigmoid for probability)
    """

    def __init__(
        self,
        input_size:    int,
        hidden_size:   int  = 64,
        num_layers:    int  = 2,
        dropout:       float = 0.3,
        bidirectional: bool  = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        direction_mult = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
            batch_first   = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size * direction_mult, hidden_size // 2)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            # Concat forward and backward final hidden states
            hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            hidden = hn[-1]
        hidden = self.dropout(hidden)
        return self.fc2(self.relu(self.fc1(hidden)))


# ── Trainer ───────────────────────────────────────────────────

class LSTMTrainer(BaseTrainer):
    """
    LSTM trainer with early stopping and cosine-annealing LR schedule.

    Usage
    -----
    trainer = LSTMTrainer()
    results = trainer.fit(feature_matrix_df)
    """

    def __init__(self, model_params: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        cfg = settings.model
        self.seq_len      = cfg.seq_len
        self.epochs       = cfg.lstm_epochs
        self.batch_size   = cfg.lstm_batch_size
        self.lr           = cfg.lstm_lr
        self.weight_decay = cfg.lstm_weight_decay
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_params = model_params or {
            "hidden_size"  : cfg.lstm_hidden_size,
            "num_layers"   : cfg.lstm_num_layers,
            "dropout"      : cfg.lstm_dropout,
            "bidirectional": cfg.lstm_bidirectional,
        }
        self.train_losses: list[float] = []
        self.val_accs:     list[float] = []

    # ── BaseTrainer implementation ────────────────────────────

    def _train_final(self) -> None:
        """Build and train the LSTM on the full training set."""
        n_features  = self.X_train.shape[1]
        self.model  = StockLSTM(input_size=n_features, **self.model_params).to(self.device)

        train_ds    = StockSequenceDataset(self.X_train, self.y_train, self.seq_len)
        val_ds      = StockSequenceDataset(self.X_test,  self.y_test,  self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        criterion  = nn.BCEWithLogitsLoss()
        optimizer  = optim.AdamW(self.model.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        log.info("Training LSTM on %d sequences (%d epochs) …",
                 len(train_ds), self.epochs)

        best_val_acc = 0.0
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x_batch).squeeze(1)
                loss   = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # ── Validate ──────────────────────────────────────
            val_acc = self._eval_accuracy(val_loader)
            self.train_losses.append(epoch_loss / max(len(train_loader), 1))
            self.val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 10 == 0:
                log.info("Epoch %3d/%d — loss=%.4f  val_acc=%.4f",
                         epoch, self.epochs, self.train_losses[-1], val_acc)

        # Restore best weights
        if best_weights:
            self.model.load_state_dict(best_weights)
        log.info("LSTM training complete. Best val accuracy: %.4f", best_val_acc)

    def _cv_fold_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Lightweight CV fold training (fewer epochs)."""
        n_features = X.shape[1]
        model = StockLSTM(input_size=n_features, **self.model_params).to(self.device)
        ds    = StockSequenceDataset(X, y, self.seq_len)
        if len(ds) == 0:
            return
        loader    = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(10):    # 10 epochs for CV speed
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch.to(self.device)).squeeze(1),
                                 y_batch.to(self.device))
                loss.backward()
                optimizer.step()
        self.model = model

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary predictions, probabilities) for a feature array."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        ds     = StockSequenceDataset(X, np.zeros(len(X)), self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        probs_list: list[float] = []
        with torch.no_grad():
            for x_batch, _ in loader:
                logits = self.model(x_batch.to(self.device)).squeeze(1)
                probs_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        probs = np.array(probs_list)
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def save(self, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or settings.paths.models_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / LSTM_MODEL_FILE
        torch.save({
            "state_dict"  : self.model.state_dict(),
            "model_params": self.model_params,
            "seq_len"     : self.seq_len,
            "feature_cols": self.feature_cols,
            "n_features"  : self.X_train.shape[1] if self.X_train is not None else None,
        }, path)
        log.info("LSTM model saved → %s", path)
        return path

    def load(self, model_path: Path) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        n_features  = checkpoint["n_features"]
        self.model  = StockLSTM(input_size=n_features,
                                **checkpoint["model_params"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.seq_len     = checkpoint.get("seq_len", self.seq_len)
        self.feature_cols = checkpoint.get("feature_cols", self.feature_cols)
        self.model.eval()
        log.info("LSTM model loaded from %s", model_path)

    # ── Internal helper ───────────────────────────────────────

    def _eval_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                logits = self.model(x.to(self.device)).squeeze(1)
                preds  = (torch.sigmoid(logits) >= 0.5).long().cpu()
                correct += (preds == y.long()).sum().item()
                total   += len(y)
        return correct / total if total > 0 else 0.0
