"""Helper functions for training transformer models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer


@dataclass(frozen=True)
class ModelSettings:
    """Settings for the transformer model."""
    block_size: int = 32
    n_layers: int = 2
    n_heads: int = 2
    n_embd: int = 64
    mlp_multiplier: int = 4


@dataclass(frozen=True)
class TrainingSettings:
    """Settings for the training process."""
    batch_size: int = 16
    train_steps: int = 300
    eval_interval: int = 50
    learning_rate: float = 3e-4


@dataclass(frozen=True)
class GenerationSettings:
    """Settings for the text generation process."""
    max_new_tokens: int = 50
    prompt_length: int = 1
    prompt_text: str | None = None


@dataclass(frozen=True)
class OutputSettings:
    """Settings for the output of the experiment."""
    create_run_dir: bool = True
    save_model: bool = True
    model_file_name: str = "model.pt"
    model_path: str | None = None
    save_generated_text: bool = True
    save_loss_history: bool = True


@dataclass(frozen=True)
class ExperimentSettings:
    """Settings for the entire experiment."""
    seed: int
    tokenizer_type: str
    device: str
    model: ModelSettings
    training: TrainingSettings
    generation: GenerationSettings
    output: OutputSettings


def parse_override_value(raw_value: str):
    """Parse a raw override value and convert it to the appropriate type."""
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        pass

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def apply_config_overrides(config: dict[str, Any],
                           overrides: list[str]) -> list[tuple[str, object]]:
    """Apply configuration overrides to a dictionary of settings."""
    applied: list[tuple[str, object]] = []
    for raw_override in overrides:
        if "=" not in raw_override:
            raise ValueError(
                f"Invalid override '{raw_override}'. Expected format key=value."
            )

        key_path, raw_value = raw_override.split("=", 1)
        key_path = key_path.strip()
        if not key_path:
            raise ValueError(
                f"Invalid override '{raw_override}'. Key cannot be empty.")

        value = parse_override_value(raw_value.strip())

        keys = [key.strip() for key in key_path.split(".") if key.strip()]
        if not keys:
            raise ValueError(
                f"Invalid override '{raw_override}'. Key cannot be empty.")

        target = config
        for key in keys[:-1]:
            existing = target.get(key)
            if existing is None:
                target[key] = {}
                existing = target[key]
            if not isinstance(existing, dict):
                raise ValueError(
                    f"Override path '{key_path}' conflicts with non-dict key '{key}'."
                )
            target = existing

        target[keys[-1]] = value
        applied.append((key_path, value))

    return applied


def load_experiment_settings(config: dict[str, Any], device: str) -> ExperimentSettings:
    """Load experiment settings from a configuration dictionary and a device string."""
    model_cfg = dict(config.get("model", {}))
    training_cfg = dict(config.get("training", {}))
    generation_cfg = dict(config.get("generation", {}))
    output_cfg = dict(config.get("output", {}))

    prompt_text = generation_cfg.get("prompt_text") or None
    if prompt_text is not None:
        prompt_text = str(prompt_text)

    model = ModelSettings(
        block_size=int(model_cfg.get("block_size", 32)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 2)),
        n_embd=int(model_cfg.get("n_embd", 64)),
        mlp_multiplier=int(model_cfg.get("mlp_multiplier", 4)),
    )
    training = TrainingSettings(
        batch_size=int(training_cfg.get("batch_size", 16)),
        train_steps=int(training_cfg.get("train_steps", 300)),
        eval_interval=int(training_cfg.get("eval_interval", 50)),
        learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
    )
    generation = GenerationSettings(
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 50)),
        prompt_length=int(generation_cfg.get("prompt_length", 1)),
        prompt_text=prompt_text,
    )
    output = OutputSettings(
        create_run_dir=bool(output_cfg.get("create_run_dir", True)),
        save_model=bool(output_cfg.get("save_model", True)),
        model_file_name=str(output_cfg.get("model_file_name", "model.pt")),
        model_path=(None if output_cfg.get("model_path")
                    is None else str(output_cfg.get("model_path"))),
        save_generated_text=bool(output_cfg.get("save_generated_text", True)),
        save_loss_history=bool(output_cfg.get("save_loss_history", True)),
    )

    return ExperimentSettings(
        seed=int(config.get("seed", 1234)),
        tokenizer_type=str(config.get("tokenizer_type", "simple_words")),
        device=device,
        model=model,
        training=training,
        generation=generation,
        output=output,
    )


def build_prompt(
    tokenizer: Tokenizer,
    token_ids: torch.Tensor,
    prompt_length: int,
    prompt_text: str | None,
    device: str,
) -> tuple[torch.Tensor, str]:
    """Builds a prompt tensor from either provided prompt text or the first 
    `prompt_length` tokens of the training stream. Returns the prompt tensor 
    and the text used to build it.
    Args:
        tokenizer: The Tokenizer object to use for encoding prompt text if provided.
        token_ids: The tensor of token IDs representing the training stream. 
                   Shape (stream_length,). Ignored if prompt_text is provided.
        prompt_length: The number of tokens to include in the prompt.
        prompt_text: The text to use for the prompt, if provided.
        device: The device on which to place the prompt tensor.
    Returns:
        A tuple containing the prompt tensor and the text used to build it."""
    if prompt_text is not None:
        prompt_ids = tokenizer.encode(prompt_text).ids
        if prompt_ids:
            prompt = torch.tensor(
                [prompt_ids], dtype=torch.long, device=device)
            return prompt, prompt_text
        print(
            f"Warning: prompt_text '{prompt_text}' encoded to zero tokens. "
            f"Falling back to first {prompt_length} token(s) of training stream."
        )

    prompt = token_ids[:prompt_length].unsqueeze(0).to(device)
    prompt_text_out = tokenizer.decode(
        prompt[0].tolist(), skip_special_tokens=False)
    return prompt, prompt_text_out


def save_generated_text(
    output_path: Path,
    checkpoint_path: Path,
    device: str,
    tokenizer_type: str,
    prompt_text: str,
    generated_text: str,
) -> None:
    """Save the generated text to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write(f"Checkpoint: {checkpoint_path}\n")
        file_obj.write(f"Device: {device}\n")
        file_obj.write(f"Tokenizer type: {tokenizer_type}\n")
        file_obj.write("Prompt:\n")
        file_obj.write(prompt_text)
        file_obj.write("\n\nGenerated sample:\n")
        file_obj.write(generated_text)
        file_obj.write("\n")


def save_loss_history(output_path: Path, loss_points: list[tuple[int, float]]) -> None:
    """Save the loss history to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write("step,loss\n")
        for step, loss_value in loss_points:
            file_obj.write(f"{step},{loss_value:.8f}\n")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    config: dict[str, Any],
    tokenizer_type: str,
    vocab_size: int,
    train_steps: int,
    final_loss: float,
) -> None:
    """Save the current state of the model and training process to a checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "train_steps": train_steps,
        "final_loss": final_loss,
    }
    torch.save(checkpoint, checkpoint_path)
