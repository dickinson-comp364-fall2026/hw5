import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STREAM_PATH = ROOT / "data" / "stream.txt"
TRAIN_SCRIPT = ROOT / "tokenization" / "train_tokenizer.py"
ENCODE_SCRIPT = ROOT / "tokenization" / "encode_stream_ids.py"
GEN_SCRIPT = ROOT / "create_data.py"


def run_python(script: Path, *args: str, seed: int = 1234) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PROJECT_SEED"] = str(seed)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TokenizationPipelineTests(unittest.TestCase):
    def test_generate_stream_is_reproducible_and_lf_only(self) -> None:
        run_python(GEN_SCRIPT, seed=42)
        first_hash = sha256_of(STREAM_PATH)

        run_python(GEN_SCRIPT, seed=42)
        second_hash = sha256_of(STREAM_PATH)

        self.assertEqual(first_hash, second_hash)
        data = STREAM_PATH.read_bytes()
        self.assertNotIn(b"\r", data)


    def test_train_tokenizer_bpe_and_simple_words(self) -> None:
        run_python(TRAIN_SCRIPT, "--tokenizer-type", "bpe", seed=42)
        run_python(TRAIN_SCRIPT, "--tokenizer-type", "simple_words", seed=42)

        self.assertTrue((ROOT / "tokenization" / "artifacts" / "bpe" / "tokenizer.json").exists())
        self.assertTrue((ROOT / "tokenization" / "artifacts" / "bpe" / "merges.txt").exists())
        self.assertTrue((ROOT / "tokenization" / "artifacts" / "simple_words" / "tokenizer.json").exists())


    def test_encode_stream_outputs_for_both_modes(self) -> None:
        run_python(ENCODE_SCRIPT, "--tokenizer-type", "bpe", seed=42)
        run_python(ENCODE_SCRIPT, "--tokenizer-type", "simple_words", seed=42)

        for mode in ("bpe", "simple_words"):
            output_dir = ROOT / "tokenization" / "encoded" / mode
            self.assertTrue((output_dir / "stream_ids.bin").exists())

            meta_path = output_dir / "stream_ids_meta.json"
            sanity_path = output_dir / "sanity_check.txt"

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["tokenizer_type"], mode)
            self.assertGreater(metadata["token_count"], 0)
            self.assertGreater(metadata["total_bytes"], 0)

            sanity = sanity_path.read_text(encoding="utf-8")
            self.assertIn("full_stream_exact_match: True", sanity)


if __name__ == "__main__":
    unittest.main()
