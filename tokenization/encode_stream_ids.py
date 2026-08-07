"""
Encode a continuous stream of text into token IDs.
"""

import argparse
import json
import struct
from pathlib import Path

from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_TEXT_PATH = PROJECT_ROOT / "data" / "stream.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode stream text into token IDs.")
    parser.add_argument(
        "-t", "--tokenizer-type",
        choices=["bpe", "simple_words"],
        default="simple_words",
        help="Tokenizer mode to load and encode with.",
    )
    return parser.parse_args()


def choose_dtype(vocab_size: int) -> tuple[str, str]:
    if vocab_size <= 65535:
        return "uint16", "<H"
    return "uint32", "<I"


def main() -> None:
    args = parse_args()
    tokenizer_path = (
        PROJECT_ROOT / "tokenization" / "artifacts" / args.tokenizer_type / "tokenizer.json"
    )
    output_dir = PROJECT_ROOT / "tokenization" / "encoded" / args.tokenizer_type
    output_bin_path = output_dir / "stream_ids.bin"
    output_meta_path = output_dir / "stream_ids_meta.json"
    sanity_report_path = output_dir / "sanity_check.txt"

    if not INPUT_TEXT_PATH.exists():
        raise FileNotFoundError(f"Input text file not found: {INPUT_TEXT_PATH}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stream_text = INPUT_TEXT_PATH.read_text(encoding="utf-8")
    encoded = tokenizer.encode(stream_text)
    token_ids = encoded.ids

    vocab_size = tokenizer.get_vocab_size()
    dtype_name, pack_fmt = choose_dtype(vocab_size)

    with output_bin_path.open("wb") as f:
        for token_id in token_ids:
            f.write(struct.pack(pack_fmt, token_id))

    preview_tokens = min(80, len(token_ids))
    prefix_ids = token_ids[:preview_tokens]
    decoded_prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    source_prefix = stream_text[: len(decoded_prefix)]
    prefix_exact_match = decoded_prefix == source_prefix
    normalized_decoded = "".join(decoded_prefix.split())
    normalized_source = "".join(source_prefix.split())
    prefix_match_ignoring_whitespace = normalized_decoded == normalized_source
    reencoded_prefix_ids = tokenizer.encode(decoded_prefix).ids
    prefix_ids_roundtrip_match = reencoded_prefix_ids == prefix_ids
    decoded_full = tokenizer.decode(token_ids, skip_special_tokens=False)
    full_stream_exact_match = decoded_full == stream_text

    metadata = {
        "input_text_path": str(INPUT_TEXT_PATH),
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_path": str(tokenizer_path),
        "output_bin_path": str(output_bin_path),
        "token_count": len(token_ids),
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "endianness": "little",
        "bytes_per_id": struct.calcsize(pack_fmt),
        "total_bytes": len(token_ids) * struct.calcsize(pack_fmt),
        "continuous_stream_mode": True,
    }
    output_meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    sanity_report = [
        f"preview_token_count: {preview_tokens}",
        f"prefix_exact_match: {prefix_exact_match}",
        f"prefix_match_ignoring_whitespace: {prefix_match_ignoring_whitespace}",
        f"prefix_ids_roundtrip_match: {prefix_ids_roundtrip_match}",
        f"full_stream_exact_match: {full_stream_exact_match}",
        f"decoded_prefix: {decoded_prefix}",
        f"source_prefix: {source_prefix}",
    ]
    sanity_report_path.write_text("\n".join(sanity_report) + "\n", encoding="utf-8")

    print("Continuous stream encoding complete.")
    print(f"Input text file: {INPUT_TEXT_PATH}")
    print(f"Tokenizer type: {args.tokenizer_type}")
    print(f"Token count: {len(token_ids)}")
    print(f"Dtype: {dtype_name}")
    print(f"Wrote: {output_bin_path}")
    print(f"Wrote: {output_meta_path}")
    print(f"Wrote: {sanity_report_path}")


if __name__ == "__main__":
    main()
