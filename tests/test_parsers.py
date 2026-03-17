"""Unit tests for modelsig.parsers modules (no network calls)."""
import json
import struct
import pytest
from modelsig.parsers.safetensors import parse_local_header
from modelsig.parsers.config import _flatten_config


def _make_safetensors_bytes(header_dict: dict) -> bytes:
    """Create a minimal well-formed safetensors byte string."""
    header_json = json.dumps(header_dict).encode("utf-8")
    length = struct.pack("<Q", len(header_json))
    return length + header_json


# ---------------------------------------------------------------------------
# safetensors.py
# ---------------------------------------------------------------------------

class TestParseLocalHeader:
    def test_basic_header(self, tmp_path):
        header = {
            "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 4096], "data_offsets": [0, 0]},
            "lm_head.weight": {"dtype": "BF16", "shape": [32000, 4096], "data_offsets": [0, 0]},
        }
        data = _make_safetensors_bytes(header)
        f = tmp_path / "model.safetensors"
        f.write_bytes(data)
        result = parse_local_header(str(f))
        assert "model.embed_tokens.weight" in result
        assert result["model.embed_tokens.weight"]["shape"] == [32000, 4096]

    def test_metadata_stripped(self, tmp_path):
        header = {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F32", "shape": [512], "data_offsets": [0, 0]},
        }
        data = _make_safetensors_bytes(header)
        f = tmp_path / "model.safetensors"
        f.write_bytes(data)
        result = parse_local_header(str(f))
        assert "__metadata__" not in result
        assert "weight" in result

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.safetensors"
        f.write_bytes(b"")
        result = parse_local_header(str(f))
        assert result == {}

    def test_bf16_dtype_preserved(self, tmp_path):
        header = {"w": {"dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 0]}}
        data = _make_safetensors_bytes(header)
        f = tmp_path / "model.safetensors"
        f.write_bytes(data)
        result = parse_local_header(str(f))
        assert result["w"]["dtype"] == "BF16"


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestFlattenConfig:
    def test_gpt2_aliases(self):
        cfg = {"model_type": "gpt2", "n_embd": 768, "n_head": 12, "n_layer": 12}
        result = _flatten_config(cfg)
        assert result["hidden_size"] == 768
        assert result["num_attention_heads"] == 12
        assert result["num_hidden_layers"] == 12

    def test_distilbert_aliases(self):
        cfg = {"model_type": "distilbert", "dim": 768, "n_heads": 12, "n_layers": 6}
        result = _flatten_config(cfg)
        assert result["hidden_size"] == 768
        assert result["num_attention_heads"] == 12
        assert result["num_hidden_layers"] == 6

    def test_t5_aliases(self):
        cfg = {"model_type": "t5", "d_model": 512, "num_layers": 6,
               "d_ff": 2048, "num_heads": 8}
        result = _flatten_config(cfg)
        assert result["hidden_size"] == 512
        assert result["num_hidden_layers"] == 6
        assert result["intermediate_size"] == 2048
        assert result["num_attention_heads"] == 8

    def test_whisper_aliases(self):
        cfg = {"model_type": "whisper", "d_model": 1024,
               "encoder_attention_heads": 16, "encoder_layers": 24}
        result = _flatten_config(cfg)
        assert result["hidden_size"] == 1024
        assert result["num_attention_heads"] == 16
        assert result["num_hidden_layers"] == 24

    def test_canonical_fields_not_overridden(self):
        cfg = {"hidden_size": 4096, "n_embd": 2048}  # canonical already set
        result = _flatten_config(cfg)
        assert result["hidden_size"] == 4096  # not overridden by alias

    def test_nested_text_config_hoisted(self):
        cfg = {
            "model_type": "llava",
            "text_config": {"hidden_size": 4096, "num_hidden_layers": 32},
        }
        result = _flatten_config(cfg)
        assert result["hidden_size"] == 4096
        assert result["num_hidden_layers"] == 32

    def test_nested_llm_config_hoisted(self):
        cfg = {
            "model_type": "video_llm",
            "llm_config": {"hidden_size": 2048, "num_attention_heads": 16},
        }
        result = _flatten_config(cfg)
        assert result["hidden_size"] == 2048

    def test_empty_config(self):
        assert _flatten_config({}) == {}

    def test_original_fields_preserved(self):
        cfg = {"hidden_size": 4096, "rope_theta": 500000.0, "custom_field": "test"}
        result = _flatten_config(cfg)
        assert result["custom_field"] == "test"
        assert result["rope_theta"] == 500000.0
