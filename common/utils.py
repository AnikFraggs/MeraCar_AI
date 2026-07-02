"""Shared helpers: reproducibility, paths, device selection, IO, and logging."""
from __future__ import annotations

import os
import random
import logging
from pathlib import Path

import numpy as np


def setup_logger(name: str = "AegisDrive", log_file: str = "aegis_drive.log", level=logging.INFO) -> logging.Logger:
    """Set up a standard logger that writes to both console and a file with timestamps."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (if available) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device, or None if torch is missing."""
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return None


def module_dir(file: str) -> Path:
    """Directory of the calling module file."""
    return Path(file).resolve().parent


def ensure_dirs(base: Path) -> tuple[Path, Path]:
    """Create (and return) the data/ and models/ dirs next to a module."""
    data = base / "data"
    models = base / "models"
    data.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return data, models