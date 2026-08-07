"""
Train a tiny causal decoder transformer using a JSON config file.
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config_utils import create_run_dir, load_config, save_config_copy
from seed_utils import get_seed, seed_everything
from transformer_helper import load_token_ids, resolve_data_paths
from transformer_model import TinyTransformerLM, resolve_device
from train_transformer_helpers import (
    apply_config_overrides,
    build_prompt,
    load_experiment_settings,
    parse_override_value,
    save_checkpoint,
    save_generated_text,
    save_loss_history,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "tiny_transformer_v1.json"


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Not enough tokens for the configured block_size.")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i: i + block_size]
                    for i in starts])  # shape: [batch_size, block_size]
    # TODO: Change the line below to create `y` as the target tokens corresponding to `x`.
    y = torch.zeros_like(x)  # placeholder, replace with correct target tokens

    return x.to(device), y.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny causal decoder transformer using a JSON config file."
    )
    parser.add_argument(
        "-c", "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to experiment config JSON.Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "-O", "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override config values using dotted keys (repeatable), for example "
            "--override training.train_steps=500 --override output.create_run_dir=false"
        ),
    )
    return parser.parse_args()


def load_training_context(args: argparse.Namespace):
    config = load_config(args.config)
    applied_overrides = apply_config_overrides(config, args.override)

    seed = int(config.get("seed", get_seed()))
    seed_everything(seed)

    device = resolve_device(str(config.get("device", "auto")))
    settings = load_experiment_settings(config, device=device)

    run_dir: Path | None = None
    if settings.output.create_run_dir:
        run_dir = create_run_dir(config)
        save_config_copy(config, run_dir)

    ids_bin_path, meta_path, tokenizer_json_path = resolve_data_paths(
        settings.tokenizer_type)    
    token_ids = load_token_ids(ids_bin_path, meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    vocab_size = int(metadata["vocab_size"])
    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))

    if len(token_ids) <= settings.model.block_size + 1:
        raise ValueError(
            f"Token stream too short ({len(token_ids)}) for block_size={settings.model.block_size}. "
            f"Need at least block_size + 2 tokens to create a single training example (input of block_size tokens + target of block_size tokens)."
        )
    if settings.generation.prompt_length < 1:
        raise ValueError("prompt_length must be at least 1.")
    if settings.generation.prompt_length > len(token_ids):
        raise ValueError(
            f"prompt_length={settings.generation.prompt_length} exceeds token count={len(token_ids)}."
        )

    # TODO: Write a description each parameter of this constructor in your own words.
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        block_size=settings.model.block_size,
        n_layers=settings.model.n_layers,
        n_heads=settings.model.n_heads,
        n_embd=settings.model.n_embd,
        mlp_multiplier=settings.model.mlp_multiplier,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.training.learning_rate)

    return {
        "config": config,
        "applied_overrides": applied_overrides,
        "run_dir": run_dir,
        "settings": settings,
        "token_ids": token_ids,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "model": model,
        "optimizer": optimizer,
    }


def print_training_summary(context: dict[str, object]) -> None:
    settings = context["settings"]
    assert hasattr(settings, "tokenizer_type")

    print(f"Config path: {context['config_path']}")
    if context["applied_overrides"]:
        print("Applied overrides:")
        for key_path, value in context["applied_overrides"]:
            print(f"  {key_path}={value}")
    if context["run_dir"] is not None:
        print(f"Run directory: {context['run_dir']}")
    print(f"Device: {settings.device}")
    print(f"Seed: {settings.seed}")
    print(f"Tokenizer type: {settings.tokenizer_type}")
    print(f"Token count: {len(context['token_ids'])}")
    print(f"Vocab size: {context['vocab_size']}")
    print(
        f"Model: layers={settings.model.n_layers}, heads={settings.model.n_heads}, "
        f"embd={settings.model.n_embd}, mlp_multiplier={settings.model.mlp_multiplier}, "
        f"block_size={settings.model.block_size}"
    )


def run_training_loop(context: dict[str, object]) -> tuple[list[tuple[int, float]], float]:
    settings = context["settings"]
    model = context["model"]
    optimizer = context["optimizer"]
    token_ids = context["token_ids"]
    assert isinstance(token_ids, torch.Tensor)

    model.train()
    loss_points: list[tuple[int, float]] = []
    last_loss_value = float("nan")

    for step in range(1, settings.training.train_steps + 1):
        xb, yb = get_batch(
            data=token_ids,
            block_size=settings.model.block_size,
            batch_size=settings.training.batch_size,
            device=settings.device,
        )

        # TODO: Add comments stating where and how this loss is computed. Describe what loss function is used and how it works in your own words.
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss_value = float(loss.item())

        if step % settings.training.eval_interval == 0 or step == 1 or step == settings.training.train_steps:
            loss_points.append((step, last_loss_value))
            print(f"step {step:4d} | loss {last_loss_value:.4f}")

    return loss_points, last_loss_value


def preview_generation(context: dict[str, object]) -> tuple[str, str]:
    settings = context["settings"]
    model = context["model"]
    tokenizer = context["tokenizer"]
    token_ids = context["token_ids"]
    assert isinstance(token_ids, torch.Tensor)

    model.eval()
    with torch.no_grad():
        prompt, prompt_text = build_prompt(
            tokenizer=tokenizer,
            token_ids=token_ids,
            prompt_length=settings.generation.prompt_length,
            prompt_text=settings.generation.prompt_text,
            device=settings.device,
        )
        generated = model.generate(
            prompt, max_new_tokens=settings.generation.max_new_tokens)
        generated_ids = generated[0].tolist()
        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=False)

    return prompt_text, generated_text


def save_run_outputs(
    context: dict[str, object],
    prompt_text: str,
    generated_text: str,
    loss_points: list[tuple[int, float]],
    last_loss_value: float,
) -> None:
    settings = context["settings"]
    config = context["config"]
    model = context["model"]
    optimizer = context["optimizer"]
    run_dir = context["run_dir"]
    vocab_size = context["vocab_size"]

    if settings.output.model_path is not None:
        checkpoint_path = Path(settings.output.model_path)
    elif run_dir is not None:
        checkpoint_path = run_dir / settings.output.model_file_name
    else:
        checkpoint_path = PROJECT_ROOT / settings.output.model_file_name

    if run_dir is not None and settings.output.save_generated_text:
        save_generated_text(
            output_path=run_dir / "generated.txt",
            checkpoint_path=checkpoint_path,
            device=settings.device,
            tokenizer_type=settings.tokenizer_type,
            prompt_text=prompt_text,
            generated_text=generated_text,
        )

    if run_dir is not None and settings.output.save_loss_history:
        save_loss_history(run_dir / "loss_history.csv", loss_points)

    if settings.output.save_model:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            config=config,
            tokenizer_type=settings.tokenizer_type,
            vocab_size=vocab_size,
            train_steps=settings.training.train_steps,
            final_loss=last_loss_value,
        )
        print(f"Saved model checkpoint: {checkpoint_path}")


def main() -> None:
    args = parse_args()

    # The returned `context` below is a dictionary. The most important items in the dictionary are:
    # -- `model`, which is the TinyTransformerLM instance to be trained;
    # -- `settings`, which contains the training settings;
    # -- `tokenizer`, which is used for tokenizing the input text;
    # -- `optimizer`, which is used to update the model parameters;
    context = load_training_context(args)
    context["config_path"] = args.config

    print_training_summary(context)

    loss_points, last_loss_value = run_training_loop(context)
    prompt_text, generated_text = preview_generation(context)

    print(f"\nInput prompt (first 100 chars): {prompt_text[:100]}")
    print("\nGenerated sample:\n")
    print(generated_text)

    save_run_outputs(context, prompt_text, generated_text,
                     loss_points, last_loss_value)


if __name__ == "__main__":
    main()
