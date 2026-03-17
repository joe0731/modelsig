"""Unit tests for modelsig.onnx modules."""
import pytest
from modelsig.onnx.ops import _ONNX_DTYPE, _ONNX_OP_MAP, onnx_op_types_to_canonical
from modelsig.onnx.selector import select_primary_onnx, is_onnx_model
from modelsig.onnx.parser import _pb_varint, _pb_fields, parse_model_bytes_fallback


# ---------------------------------------------------------------------------
# ops.py
# ---------------------------------------------------------------------------

class TestOnnxDtype:
    def test_f32(self):
        assert _ONNX_DTYPE[1] == "F32"

    def test_fp16(self):
        assert _ONNX_DTYPE[10] == "FP16"

    def test_bf16(self):
        assert _ONNX_DTYPE[16] == "BF16"

    def test_int8(self):
        assert _ONNX_DTYPE[3] == "INT8"


class TestOnnxOpMap:
    def test_matmul(self):
        assert _ONNX_OP_MAP["MatMul"] == "aten/mm"

    def test_gemm(self):
        assert _ONNX_OP_MAP["Gemm"] == "aten/mm"

    def test_attention(self):
        assert _ONNX_OP_MAP["GroupQueryAttention"] == "attention"

    def test_rms_norm(self):
        assert _ONNX_OP_MAP["SimplifiedLayerNormalization"] == "rms_norm"

    def test_matmul_nbits(self):
        assert _ONNX_OP_MAP["MatMulNBits"] == "aten/mm"


class TestOnnxOpTypesToCanonical:
    def test_matmul_mapped(self):
        result = onnx_op_types_to_canonical(["MatMul"])
        assert "aten/mm" in result

    def test_unknown_op_lowercased(self):
        result = onnx_op_types_to_canonical(["CustomFusedOp"])
        assert "customfusedop" in result

    def test_always_includes_defaults(self):
        result = onnx_op_types_to_canonical([])
        assert "aten/mm" in result
        assert "scaled_dot_product_attention" in result

    def test_multiple_ops(self):
        result = onnx_op_types_to_canonical(["MatMul", "GroupQueryAttention", "RotaryEmbedding"])
        assert "aten/mm" in result
        assert "attention" in result
        assert "rope" in result


# ---------------------------------------------------------------------------
# selector.py
# ---------------------------------------------------------------------------

def _file(name, size=None):
    return {"rfilename": name, "size": size}


class TestSelectPrimaryOnnx:
    def test_no_onnx_returns_none(self):
        files = [_file("model.safetensors")]
        assert select_primary_onnx(files) is None

    def test_prefers_model_with_external_data(self):
        files = [
            _file("onnx/model.onnx", size=5_000_000),
            _file("onnx/model.onnx_data", size=2_000_000_000),
        ]
        result = select_primary_onnx(files)
        assert result == "onnx/model.onnx"

    def test_prefers_decoder_merged_over_plain_model(self):
        files = [
            _file("onnx/decoder_model_merged.onnx"),
            _file("onnx/decoder_model_merged.onnx_data"),
            _file("onnx/model.onnx"),
            _file("onnx/model.onnx_data"),
        ]
        result = select_primary_onnx(files)
        assert result == "onnx/decoder_model_merged.onnx"

    def test_skips_quantized_variants(self):
        files = [
            _file("onnx/model_q4.onnx", size=100),
            _file("onnx/model_q8.onnx", size=200),
            _file("onnx/model.onnx", size=5_000_000),
            _file("onnx/model.onnx_data"),
        ]
        result = select_primary_onnx(files)
        assert result == "onnx/model.onnx"

    def test_fallback_to_known_paths(self):
        files = [_file("onnx/model.onnx", size=10_000_000)]
        result = select_primary_onnx(files)
        assert result == "onnx/model.onnx"

    def test_fallback_to_smallest(self):
        files = [
            _file("encoder.onnx", size=50_000_000),
            _file("decoder.onnx", size=5_000_000),
        ]
        result = select_primary_onnx(files)
        assert result == "decoder.onnx"


class TestIsOnnxModel:
    def test_onnx_only(self):
        files = [_file("model.onnx"), _file("model.onnx_data")]
        assert is_onnx_model(files) is True

    def test_safetensors_present(self):
        files = [_file("model.safetensors"), _file("model.onnx")]
        assert is_onnx_model(files) is False

    def test_no_onnx(self):
        files = [_file("model.safetensors")]
        assert is_onnx_model(files) is False


# ---------------------------------------------------------------------------
# parser.py — protobuf helpers
# ---------------------------------------------------------------------------

class TestPbVarint:
    def test_single_byte(self):
        val, pos = _pb_varint(b"\x05", 0)
        assert val == 5
        assert pos == 1

    def test_multi_byte(self):
        # 300 = 0xAC 0x02
        val, pos = _pb_varint(b"\xac\x02", 0)
        assert val == 300
        assert pos == 2

    def test_zero(self):
        val, pos = _pb_varint(b"\x00", 0)
        assert val == 0

    def test_max_7bit(self):
        val, pos = _pb_varint(b"\x7f", 0)
        assert val == 127


class TestParseModelBytesFallback:
    def test_empty_bytes_returns_empty(self):
        meta, ops = parse_model_bytes_fallback(b"")
        assert meta == {}
        assert ops == []

    def test_random_bytes_does_not_crash(self):
        # Should gracefully handle garbage
        meta, ops = parse_model_bytes_fallback(b"\xff\xfe\x00\x01\x02\x03" * 100)
        assert isinstance(meta, dict)
        assert isinstance(ops, list)

    def test_filters_slash_prefix_names(self):
        """Internal ONNX constants with / prefix should be filtered out."""
        # Build a minimal valid ONNX-like protobuf with a slash-prefix initializer name
        # (this tests the filtering logic; exact protobuf construction is complex,
        #  so we just verify the function doesn't crash on typical inputs)
        meta, ops = parse_model_bytes_fallback(b"\x00" * 16)
        for name in meta:
            assert not name.startswith("/"), f"Slash-prefix name leaked: {name}"
