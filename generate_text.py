"""Generate text from a saved tiny transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from transformer_helper import find_latest_checkpoint, load_token_ids, resolve_data_paths
from transformer_model import TinyTransformerLM, resolve_device
from train_transformer_helpers import build_prompt, save_generated_text


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MAX_NEW_TOKENS = 50


def get_first_token_top_probabilities(model, tokenizer, prompt, top_k=10):
    ctx_tokens = prompt[:, -model.block_size:]

    # TODO: Replace the right hand side of the assignment statement to compute logits from ctx_tokens.
    logits, _ = None, None
    # TODO: Replace the right hand side of the assignment statement to extract the logits for the next token from the logits tensor. The logits tensor has shape (batch_size, sequence_length, vocab_size), and you want to extract the logits for the last token in the sequence. The resulting next_token_logits tensor should have shape (batch_size, vocab_size).
    next_token_logits = None


    probs = torch.softmax(next_token_logits, dim=-1)
    k = min(int(top_k), int(probs.shape[-1]))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    rows = []
    for token_id, prob in zip(top_ids[0].tolist(), top_probs[0].tolist()):
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        rows.append((int(token_id), token_text, float(prob)))
    return rows


def generate_text_from_prompt(args, model, tokenizer, prompt, max_new_tokens):
    with torch.no_grad():
        first_token_top10 = None
        if args.show_first_token_top10 and max_new_tokens > 0:
            first_token_top10 = get_first_token_top_probabilities(
                model, tokenizer, prompt, top_k=10)

        # TODO: (Step 5) Replace the right hand side of the assignment statement with code that will generate new tokens from the model, starting with the prompt and producing at most max_new_tokens tokens.
        generated = []  # Replace this line

        generated_ids = generated[0].tolist()
        if args.token_separator == "":
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=False)
        else:
            generated_text = args.token_separator.join(
                tokenizer.decode([tid], skip_special_tokens=False)
                for tid in generated_ids
            )
    return generated_text, first_token_top10


def setup_tokenizer_and_prompt(args, device, config, ids_path, meta_path, tokenizer_json_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
    token_ids = load_token_ids(ids_path, meta_path)
    prompt_length = int(config.get("generation", {}).get("prompt_length", 1))
    prompt_text = args.prompt
    prompt, prompt_text = build_prompt(
        tokenizer, token_ids, prompt_length, prompt_text, device)

    generation_cfg = dict(config.get("generation", {}))
    max_new_tokens = int(args.max_new_tokens) if args.max_new_tokens is not None else int(
        generation_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    return tokenizer, prompt_text, prompt, max_new_tokens


def build_and_load_model(device, checkpoint, config, vocab_size):
    model_cfg = dict(config.get("model", {}))
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        block_size=int(model_cfg.get("block_size", 32)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 2)),
        n_embd=int(model_cfg.get("n_embd", 64)),
        mlp_multiplier=int(model_cfg.get("mlp_multiplier", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_and_report_generation(args, checkpoint_path, device, tokenizer_type, prompt_text, generated_text, first_token_top10):
    # Print summary
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Tokenizer type: {tokenizer_type}")
    print(f"Prompt (first 100 chars): {prompt_text[:100]}")
    if first_token_top10:
        print("\nTop 10 next-token probabilities for the first generated token:")
        for rank, (token_id, token_text, prob) in enumerate(first_token_top10, start=1):
            token_display = repr(token_text)
            print(f"{rank:2d}. id={token_id:<6d} prob={prob:.6f} ({prob * 100:6.2f}%) token={token_display}")
    print("\nGenerated sample:\n")
    print(generated_text)

    # Save generated text to file
    output_path = Path(args.output_file) if args.output_file is not None else checkpoint_path.with_name(
        "generated_sample.txt")
    save_generated_text(output_path, checkpoint_path, str(
        device), tokenizer_type, prompt_text, generated_text)
    print(f"\nSaved generated sample: {output_path}")


def load_tokenizer_resources(args, checkpoint, config):
    # Determine tokenizer type (CLI override > checkpoint > config default)
    tokenizer_type = args.tokenizer_type or checkpoint.get("tokenizer_type")
    if tokenizer_type is None:
        tokenizer_type = str(config.get("tokenizer_type", "simple_words"))
    tokenizer_type = str(tokenizer_type)

    # Locate tokenizer artifacts and vocab metadata
    ids_path, meta_path, tokenizer_json_path = resolve_data_paths(
        tokenizer_type)
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(
            f"Tokenizer artifact not found: {tokenizer_json_path}. Train tokenizer artifacts for this tokenizer_type first.")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    vocab_size = int(checkpoint.get("vocab_size", metadata["vocab_size"]))
    return tokenizer_type, ids_path, meta_path, tokenizer_json_path, vocab_size


def load_checkpoint_and_config(args):
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else find_latest_checkpoint(
        RUNS_DIR, args.model_file_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint + config
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    return checkpoint_path, device, checkpoint, config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a saved tiny transformer checkpoint.")
    parser.add_argument("-c", "--checkpoint", default=None,
                        help="Path to checkpoint file. If omitted, latest runs/*/model.pt is used.")
    parser.add_argument("-m", "--model-file-name", default="model.pt",
                        help="Model filename to look for in runs/ when --checkpoint is omitted.")
    parser.add_argument("-p", "--prompt", default=None,
                        help="Prompt text. If omitted, uses the first token from training stream.")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=None,
                        help=f"Number of new tokens to generate. Defaults to {DEFAULT_MAX_NEW_TOKENS}.")
    parser.add_argument("-t", "--tokenizer-type", default=None,
                        help="Override tokenizer type (otherwise read from checkpoint).")
    parser.add_argument("-d", "--device", default="auto",
                        help="Device to use: auto, cpu, or cuda.")
    parser.add_argument("-o", "--output-file", default=None, help=(
        "Path to save generated text. If omitted, saves next to the checkpoint " "as generated_sample.txt."),)
    parser.add_argument("-s", "--token-separator", default="",
                        help="Separator to insert between each token in the output (e.g. '|'). "
                             "If omitted, tokens are decoded as a single sequence.")
    parser.add_argument("-S", "--show-first-token-top10", action="store_true",
                        help="Print the top 10 token probabilities for the first generated token.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path, device, checkpoint, config = load_checkpoint_and_config(
        args)

    tokenizer_type, ids_path, meta_path, tokenizer_json_path, vocab_size = load_tokenizer_resources(
        args, checkpoint, config)

    model = build_and_load_model(device, checkpoint, config, vocab_size)
    model.eval()

    tokenizer, prompt_text, prompt, max_new_tokens = setup_tokenizer_and_prompt(
        args, device, config, ids_path, meta_path, tokenizer_json_path)

    generated_text, first_token_top10 = generate_text_from_prompt(
        args, model, tokenizer, prompt, max_new_tokens)

    save_and_report_generation(
        args, checkpoint_path, device, tokenizer_type, prompt_text, generated_text, first_token_top10)


if __name__ == "__main__":
    main()
