from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config.config import settings
from src.model_trainer.lstm_trainer import StockLSTM, StockSequenceDataset
from utils.logging_utils import get_logger

log = get_logger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    log.warning("Optuna not installed. LSTM tuning unavailable.")


class LSTMTuner:
    """
    Hyperparameter search for the LSTM model using Optuna.

    Searches over:
      hidden_size, num_layers, dropout, learning_rate,
      weight_decay, seq_len, batch_size

    Usage
    -----
    tuner = LSTMTuner(n_trials=30, epochs_per_trial=20)
    best_params = tuner.tune(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        n_trials: int        = 30,
        epochs_per_trial: int = 15,
        random_seed: int | None = None,
    ):
        if not _OPTUNA_AVAILABLE:
            raise ImportError("Install optuna: pip install optuna")
        self.n_trials         = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.random_seed      = random_seed or settings.model.random_seed
        self.device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params: dict = {}
        self.study            = None

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> dict:
        """
        Run Optuna search and return best hyperparameters dict.

        Parameters
        ----------
        X_train / y_train : scaled training features and labels
        X_val   / y_val   : scaled validation features and labels

        Returns
        -------
        dict with keys: hidden_size, num_layers, dropout, learning_rate,
                        weight_decay, seq_len, batch_size
        """
        log.info("Starting Optuna LSTM tuning (%d trials, %d epochs each) …",
                 self.n_trials, self.epochs_per_trial)

        def objective(trial: optuna.Trial) -> float:
            hidden_size   = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
            num_layers    = trial.suggest_int("num_layers", 1, 3)
            dropout       = trial.suggest_float("dropout", 0.1, 0.5)
            lr            = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            seq_len       = trial.suggest_categorical("seq_len", [5, 10, 15, 20])
            batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64])
            bidirectional = trial.suggest_categorical("bidirectional", [True, False])

            train_ds = StockSequenceDataset(X_train, y_train, seq_len)
            val_ds   = StockSequenceDataset(X_val,   y_val,   seq_len)

            if len(train_ds) == 0 or len(val_ds) == 0:
                return 0.0

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

            n_features = X_train.shape[1]
            model = StockLSTM(
                input_size    = n_features,
                hidden_size   = hidden_size,
                num_layers    = num_layers,
                dropout       = dropout,
                bidirectional = bidirectional,
            ).to(self.device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Quick training
            model.train()
            for _ in range(self.epochs_per_trial):
                for x_b, y_b in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x_b.to(self.device)).squeeze(1),
                                     y_b.to(self.device))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # Evaluate
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x_b, y_b in val_loader:
                    logits = model(x_b.to(self.device)).squeeze(1)
                    preds  = (torch.sigmoid(logits) >= 0.5).long().cpu()
                    correct += (preds == y_b.long()).sum().item()
                    total   += len(y_b)

            return correct / total if total > 0 else 0.0

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = self.study.best_params
        log.info("Optuna LSTM best accuracy: %.4f | params: %s",
                 self.study.best_value, self.best_params)
        return self.best_params
