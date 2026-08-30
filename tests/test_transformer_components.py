"""
Unit tests for transformer_model, transformer_io, and train_transformer utilities.
Run with:
    python -m unittest discover -s tests -p "test_*.py" -v
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_transformer import apply_config_overrides, get_batch, parse_override_value
from transformer_helper import find_latest_checkpoint, load_token_ids, resolve_data_paths
from transformer_model import TinyTransformerLM, resolve_device


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------
class ResolveDeviceTests(unittest.TestCase):
    def test_cpu_passthrough(self) -> None:
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_returns_valid_device(self) -> None:
        result = resolve_device("auto")
        self.assertIn(result, ("cpu", "cuda"))

    def test_leading_whitespace_is_stripped(self) -> None:
        self.assertEqual(resolve_device("  cpu  "), "cpu")

    def test_cuda_falls_back_to_cpu_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available; fallback not triggered.")
        result = resolve_device("cuda")
        self.assertEqual(result, "cpu")


# ---------------------------------------------------------------------------
# TinyTransformerLM — forward pass shapes
# ---------------------------------------------------------------------------
class TinyTransformerLMShapeTests(unittest.TestCase):
    def _make_model(
        self,
        vocab_size: int = 16,
        block_size: int = 8,
        n_layers: int = 1,
        n_heads: int = 2,
        n_embd: int = 8,
        mlp_multiplier: int = 2,
    ) -> TinyTransformerLM:
        return TinyTransformerLM(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_embd=n_embd,
            mlp_multiplier=mlp_multiplier,
        )

    def test_logits_shape(self) -> None:
        model = self._make_model()
        batch, seq = 4, 6
        idx = torch.randint(0, 16, (batch, seq))
        logits, loss = model(idx)
        self.assertEqual(logits.shape, (batch, seq, 16))
        self.assertIsNone(loss)

    def test_loss_is_scalar_when_targets_provided(self) -> None:
        model = self._make_model()
        idx = torch.randint(0, 16, (4, 6))
        targets = torch.randint(0, 16, (4, 6))
        _, loss = model(idx, targets)
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, torch.Size([]))  # scalar

    def test_loss_decreases_after_training_step(self) -> None:
        torch.manual_seed(0)
        model = self._make_model(vocab_size=16, block_size=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        idx = torch.randint(0, 16, (4, 8))
        targets = torch.randint(0, 16, (4, 8))

        _, loss_before = model(idx, targets)
        assert loss_before is not None
        first_loss = float(loss_before.item())

        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(idx, targets)
            assert loss is not None
            loss.backward()
            optimizer.step()

        _, loss_after = model(idx, targets)
        assert loss_after is not None
        self.assertLess(float(loss_after.item()), first_loss)

    def test_generate_grows_sequence(self) -> None:
        model = self._make_model()
        model.eval()
        prompt = torch.randint(0, 16, (1, 3))
        generated = model.generate(prompt, max_new_tokens=5)
        self.assertEqual(generated.shape, (1, 8))  # 3 + 5

    def test_generate_with_short_prompt(self) -> None:
        model = self._make_model(block_size=8)
        model.eval()
        prompt = torch.randint(0, 16, (1, 1))
        generated = model.generate(prompt, max_new_tokens=4)
        self.assertEqual(generated.shape, (1, 5))

    def test_causal_mask_shape(self) -> None:
        model = self._make_model(block_size=8)
        block = model.blocks[0]
        self.assertEqual(block.causal_mask.shape, (8, 8))
        # Upper triangle (excluding diagonal) should be True (masked)
        self.assertTrue(block.causal_mask[0, 1].item())
        # Lower triangle + diagonal should be False (not masked)
        self.assertFalse(block.causal_mask[1, 0].item())
        self.assertFalse(block.causal_mask[0, 0].item())


# ---------------------------------------------------------------------------
# get_batch
# ---------------------------------------------------------------------------
class GetBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.data = torch.arange(100, dtype=torch.long)

    def test_output_shapes(self) -> None:
        x, y = get_batch(self.data, block_size=8, batch_size=4, device="cpu")
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))

    def test_y_is_x_shifted_by_one(self) -> None:
        # With sequential data x[i]+1 == y[i] for all positions
        data = torch.arange(200, dtype=torch.long)
        x, y = get_batch(data, block_size=16, batch_size=8, device="cpu")
        self.assertTrue(torch.all(x + 1 == y))

    def test_raises_if_data_too_short(self) -> None:
        with self.assertRaises(ValueError):
            get_batch(torch.arange(5, dtype=torch.long), block_size=8, batch_size=2, device="cpu")


# ---------------------------------------------------------------------------
# parse_override_value
# ---------------------------------------------------------------------------
class ParseOverrideValueTests(unittest.TestCase):
    def test_integer(self) -> None:
        self.assertEqual(parse_override_value("42"), 42)
        self.assertIsInstance(parse_override_value("42"), int)

    def test_float(self) -> None:
        self.assertAlmostEqual(parse_override_value("3.14"), 3.14)

    def test_true_false(self) -> None:
        self.assertIs(parse_override_value("true"), True)
        self.assertIs(parse_override_value("false"), False)
        self.assertIs(parse_override_value("True"), True)

    def test_null(self) -> None:
        self.assertIsNone(parse_override_value("null"))

    def test_string_passthrough(self) -> None:
        self.assertEqual(parse_override_value("simple_words"), "simple_words")

    def test_json_list(self) -> None:
        result = parse_override_value("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# apply_config_overrides
# ---------------------------------------------------------------------------
class ApplyConfigOverridesTests(unittest.TestCase):
    def test_top_level_override(self) -> None:
        config = {"seed": 1234}
        apply_config_overrides(config, ["seed=99"])
        self.assertEqual(config["seed"], 99)

    def test_nested_override(self) -> None:
        config = {"training": {"train_steps": 100}}
        apply_config_overrides(config, ["training.train_steps=500"])
        self.assertEqual(config["training"]["train_steps"], 500)

    def test_creates_missing_nested_key(self) -> None:
        config: dict = {}
        apply_config_overrides(config, ["model.n_layers=4"])
        self.assertEqual(config["model"]["n_layers"], 4)

    def test_returns_applied_list(self) -> None:
        config: dict = {"x": 1}
        applied = apply_config_overrides(config, ["x=2"])
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0], ("x", 2))

    def test_raises_on_missing_equals(self) -> None:
        with self.assertRaises(ValueError):
            apply_config_overrides({}, ["no_equals_sign"])

    def test_raises_on_empty_key(self) -> None:
        with self.assertRaises(ValueError):
            apply_config_overrides({}, ["=value"])

    def test_multiple_overrides(self) -> None:
        config = {"a": 1, "b": 2}
        apply_config_overrides(config, ["a=10", "b=20"])
        self.assertEqual(config["a"], 10)
        self.assertEqual(config["b"], 20)


# ---------------------------------------------------------------------------
# transformer_io: load_token_ids
# ---------------------------------------------------------------------------
class LoadTokenIdsTests(unittest.TestCase):
    def _write_ids(self, tmp: Path, ids: list[int], dtype: str) -> tuple[Path, Path]:
        fmt = "H" if dtype == "uint16" else "I"
        ids_path = tmp / "stream_ids.bin"
        with ids_path.open("wb") as f:
            for i in ids:
                f.write(struct.pack(f"<{fmt}", i))
        meta_path = tmp / "stream_ids_meta.json"
        meta_path.write_text(
            json.dumps({"dtype": dtype, "token_count": len(ids)}), encoding="utf-8"
        )
        return ids_path, meta_path

    def test_uint16_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ids_path, meta_path = self._write_ids(tmp, [0, 1, 2, 3], "uint16")
            tensor = load_token_ids(ids_path, meta_path)
            self.assertEqual(tensor.tolist(), [0, 1, 2, 3])
            self.assertEqual(tensor.dtype, torch.long)

    def test_uint32_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ids_path, meta_path = self._write_ids(tmp, [100, 200, 300], "uint32")
            tensor = load_token_ids(ids_path, meta_path)
            self.assertEqual(tensor.tolist(), [100, 200, 300])

    def test_missing_ids_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            meta_path = tmp / "meta.json"
            meta_path.write_text(json.dumps({"dtype": "uint16"}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_token_ids(tmp / "missing.bin", meta_path)

    def test_unsupported_dtype_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ids_path = tmp / "stream_ids.bin"
            ids_path.write_bytes(b"")
            meta_path = tmp / "meta.json"
            meta_path.write_text(json.dumps({"dtype": "float32"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_token_ids(ids_path, meta_path)


# ---------------------------------------------------------------------------
# transformer_io: find_latest_checkpoint
# ---------------------------------------------------------------------------
class FindLatestCheckpointTests(unittest.TestCase):
    def test_finds_most_recent_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            runs = Path(tmp_str)
            old = runs / "2026-01-01-run_a"
            new = runs / "2026-06-01-run_b"
            old.mkdir()
            new.mkdir()
            # Touch new after old so mtime is later
            (old / "model.pt").write_bytes(b"old")
            import time; time.sleep(0.01)
            (new / "model.pt").write_bytes(b"new")

            result = find_latest_checkpoint(runs, "model.pt")
            self.assertEqual(result, new / "model.pt")

    def test_raises_if_no_runs_dir(self) -> None:
        with self.assertRaises(FileNotFoundError):
            find_latest_checkpoint(Path("/nonexistent/runs"), "model.pt")

    def test_raises_if_no_checkpoint_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            runs = Path(tmp_str)
            (runs / "some_run").mkdir()
            with self.assertRaises(FileNotFoundError):
                find_latest_checkpoint(runs, "model.pt")

    def test_skips_runs_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            runs = Path(tmp_str)
            empty = runs / "2026-01-01-empty"
            has_ckpt = runs / "2026-01-02-has_ckpt"
            empty.mkdir()
            import time; time.sleep(0.01)
            has_ckpt.mkdir()
            (has_ckpt / "model.pt").write_bytes(b"ckpt")

            result = find_latest_checkpoint(runs, "model.pt")
            self.assertEqual(result, has_ckpt / "model.pt")


# ---------------------------------------------------------------------------
# transformer_io: resolve_data_paths
# ---------------------------------------------------------------------------
class ResolveDataPathsTests(unittest.TestCase):
    def test_returns_three_paths(self) -> None:
        ids_path, meta_path, tokenizer_path = resolve_data_paths("simple_words")
        self.assertIsInstance(ids_path, Path)
        self.assertIsInstance(meta_path, Path)
        self.assertIsInstance(tokenizer_path, Path)

    def test_paths_contain_tokenizer_type(self) -> None:
        ids_path, meta_path, tokenizer_path = resolve_data_paths("bpe")
        self.assertIn("bpe", str(ids_path))
        self.assertIn("bpe", str(meta_path))
        self.assertIn("bpe", str(tokenizer_path))


if __name__ == "__main__":
    unittest.main()
