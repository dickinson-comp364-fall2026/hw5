"""Train a tokenizer for a continuous stream of text.

The tokenizer can be created in one of two modes: bpe or simple_words. 
"""

import argparse
import re
from pathlib import Path
import sys

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = PROJECT_ROOT / "data" / "stream.txt"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TARGET_VOCAB_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tokenizer artifacts.")
    parser.add_argument(
        "-t", "--tokenizer-type",
        choices=["bpe", "simple_words"],
        default="simple_words",
        help="Tokenizer mode to train. Default: simple_words",
    )
    parser.add_argument(
        "-i", "--input-path",
        default=str(INPUT_PATH),
        help=f"Path to the input text stream file. Default: {INPUT_PATH}",
    )
    parser.add_argument(
        "-a", "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help=f"Directory to write tokenizer artifacts. Default: {ARTIFACTS_DIR}",
    )
    parser.add_argument(
        "-v", "--target-vocab-size",
        type=int,
        default=TARGET_VOCAB_SIZE,
        help=f"Target vocabulary size for BPE tokenizer. Default: {TARGET_VOCAB_SIZE}",
    )
    return parser.parse_args()


def build_simple_words_tokenizer(stream_text: str) -> Tokenizer:
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "\n": 2,
        " ": 3,
        ".": 4,
    }

    word_tokens = sorted({tok for tok in re.split(r"[ .\n]+", stream_text) if tok})
    for token in word_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"( |\.|\n)"),
        behavior="isolated",
    )
    tokenizer.decoder = decoders.Fuse()
    return tokenizer


def build_bpe_tokenizer(input_path: Path, target_vocab_size: int) -> tuple[Tokenizer, int]:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"( |\.|\n)"),
        behavior="isolated",
    )
    tokenizer.decoder = decoders.Fuse()

    effective_vocab_size = target_vocab_size

    trainer = trainers.BpeTrainer(
        vocab_size=effective_vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]"],
    )
    tokenizer.train(files=[str(input_path)], trainer=trainer)
    return tokenizer, effective_vocab_size


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    # if not input_path.is_absolute():
    #     input_path = (Path.cwd() / input_path).resolve()

    artifacts_dir = Path(args.artifacts_dir)

    if not input_path.exists():
        msg = f"Input file not found: {input_path}."
        raise FileNotFoundError(msg)

    output_dir = artifacts_dir / args.tokenizer_type
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_text = input_path.read_text(encoding="utf-8")

    if args.tokenizer_type == "simple_words":
        tokenizer = build_simple_words_tokenizer(stream_text)
        tokenizer.model.save(str(output_dir))
        effective_vocab_size = tokenizer.get_vocab_size()
    else:
        tokenizer, effective_vocab_size = build_bpe_tokenizer(input_path, args.target_vocab_size)
        tokenizer.model.save(str(output_dir))

    tokenizer.save(str(output_dir / "tokenizer.json"))

    encoded = tokenizer.encode(stream_text)

    sample_report = [
        "mode: continuous_stream",
        f"tokenizer_type: {args.tokenizer_type}",
        f"input_chars: {len(stream_text)}",
        f"token_count: {len(encoded.ids)}",
        f"first_30_tokens: {encoded.tokens[:30]}",
        f"first_30_ids: {encoded.ids[:30]}",
    ]
    (output_dir / "sample_encoding.txt").write_text(
        "\n".join(sample_report) + "\n", encoding="utf-8"
    )

    print("Tokenizer trained.")
    print(f"Input: {input_path}")
    print(f"Tokenizer type: {args.tokenizer_type}")
    print(f"Configured vocab size: {effective_vocab_size}")
    print(f"Artifacts written to: {output_dir}")
    if args.tokenizer_type == "bpe":
        print("Files: vocab.json, merges.txt, tokenizer.json, sample_encoding.txt")
    else:
        print("Files: vocab.json, tokenizer.json, sample_encoding.txt")


if __name__ == "__main__":
    main()
