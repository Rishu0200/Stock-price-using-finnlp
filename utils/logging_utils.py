# ──────────────────────────────────────────────────────────────
#  utils/logging_utils.py
#  Centralised logging setup. Call get_logger(__name__) in every
#  module to get a consistently formatted logger.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None, level: str = "INFO") -> logging.Logger:
    """
    Return a logger with:
    - StreamHandler (stdout) always attached
    - FileHandler attached when log_file is supplied
    - Consistent [LEVEL] module_name: message format

    Parameters
    ----------
    name     : typically __name__ of the calling module
    log_file : optional path to write logs to disk
    level    : logging level string (INFO, DEBUG, WARNING, ERROR)
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # ── File handler (optional) ───────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
