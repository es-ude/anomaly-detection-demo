from pathlib import Path

import torch


def save_model(model: torch.nn.Module, destination: Path) -> None:
    torch.save(model.state_dict(), destination)


def load_model(model: torch.nn.Module, checkpoint_file: Path) -> None:
    state_dict = torch.load(checkpoint_file, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
