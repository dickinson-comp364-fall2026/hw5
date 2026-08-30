"""Helper functions for working with transformer models."""

from __future__ import annotations

import json
from array import array
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

def resolve_device(device_setting: str) -> str:
    """Resolve the device to use for training."""
    normalized = device_setting.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print("Requested device 'cuda' is unavailable. Falling back to cpu.")
        return "cpu"
    return normalized

def resolve_data_paths(tokenizer_type: str) -> tuple[Path, Path, Path]:
    """Resolve the data paths for the specified tokenizer type."""
    encoded_dir = PROJECT_ROOT / "tokenization" / "encoded" / tokenizer_type
    artifacts_dir = PROJECT_ROOT / "tokenization" / "artifacts" / tokenizer_type
    ids_bin_path = encoded_dir / "stream_ids.bin"
    meta_path = encoded_dir / "stream_ids_meta.json"
    tokenizer_json_path = artifacts_dir / "tokenizer.json"
    return ids_bin_path, meta_path, tokenizer_json_path


def load_token_ids(ids_path: Path, meta_path: Path) -> torch.Tensor:
    """Load token IDs from a binary file and return them as a PyTorch tensor."""
    if not ids_path.exists():
        raise FileNotFoundError(f"Token ID file not found: {ids_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    dtype_name = metadata["dtype"]

    if dtype_name == "uint16":
        arr = array("H")
    elif dtype_name == "uint32":
        arr = array("I")
    else:
        raise ValueError(f"Unsupported dtype in metadata: {dtype_name}")

    with ids_path.open("rb") as file_obj:
        arr.frombytes(file_obj.read())

    return torch.tensor(arr, dtype=torch.long)


def find_latest_checkpoint(runs_dir: Path, model_file_name: str) -> Path:
    """Find the latest checkpoint in the runs directory."""
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"Runs directory not found: {runs_dir}. Provide --checkpoint explicitly."
        )

    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        candidate = run_dir / model_file_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No checkpoint named '{model_file_name}' found under {runs_dir}. "
        "Provide --checkpoint explicitly."
    )