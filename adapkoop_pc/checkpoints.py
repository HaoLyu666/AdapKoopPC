from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CheckpointSet:
    ds_encoder: Path
    state_encoder: Path
    koopman: Path
    encoder: Path


def checkpoint_set(checkpoint_dir: str | Path, epoch: str | int) -> CheckpointSet:
    checkpoint_dir = Path(checkpoint_dir)
    epoch = str(epoch)
    paths = CheckpointSet(
        ds_encoder=checkpoint_dir / f"epoch{epoch}_ds.tar",
        state_encoder=checkpoint_dir / f"epoch{epoch}_st.tar",
        koopman=checkpoint_dir / f"epoch{epoch}_k.tar",
        encoder=checkpoint_dir / f"epoch{epoch}_e.tar",
    )
    missing = [str(path) for path in paths.__dict__.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files: " + ", ".join(missing))
    return paths


def torch_load_state(path: str | Path, device: str | torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state(module: torch.nn.Module, path: str | Path, device: str | torch.device) -> None:
    module.load_state_dict(torch_load_state(path, device))


def infer_hidden_size(checkpoint_dir: str | Path, epoch: str | int) -> int:
    """Infer the Koopman hidden dimension from a checkpoint set."""

    paths = checkpoint_set(checkpoint_dir, epoch)
    state_dict = torch_load_state(paths.state_encoder, "cpu")
    weight = state_dict.get("linear1.weight")
    if weight is None:
        raise ValueError(f"Cannot infer hidden size from {paths.state_encoder}")
    return int(weight.shape[0])
