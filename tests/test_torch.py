"""Unit tests for modelsig.torch modules (fx_trace and hooks).

These tests are designed to be robust in environments with or without
a full GPU/transformers setup. All model-loading calls are mocked.
"""
import sys
import types
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — build minimal mock torch + transformers environment
# ---------------------------------------------------------------------------

def _make_mock_torch():
    """Create a minimal mock of the torch module."""
    torch = MagicMock(name="torch")
    # torch.Tensor
    tensor = MagicMock(name="Tensor")
    tensor.__class__ = MagicMock()
    torch.Tensor = tensor.__class__
    # torch.device context manager
    torch.device.return_value.__enter__ = lambda s: s
    torch.device.return_value.__exit__ = MagicMock(return_value=False)
    # torch.long, torch.float32
    torch.long = 0
    torch.float32 = 1
    # torch.zeros / torch.empty
    mock_input = MagicMock(spec=["shape", "__class__"])
    mock_input.shape = [1, 8]
    torch.zeros.return_value = mock_input
    torch.empty.return_value = mock_input
    torch.no_grad.return_value.__enter__ = lambda s: s
    torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return torch


def _make_mock_model():
    """Create a minimal mock model with named_modules() and register_forward_hook()."""
    model = MagicMock(name="AutoModelForCausalLM_instance")
    model.eval.return_value = model
    # named_modules returns a small list of (name, module) pairs
    sub1 = MagicMock(name="layer0")
    sub2 = MagicMock(name="layer1")
    model.named_modules.return_value = [("layer0", sub1), ("layer1", sub2)]
    # forward call (model(dummy)) does nothing
    model.return_value = MagicMock()
    # register_forward_hook returns a handle
    sub1.register_forward_hook.return_value = MagicMock()
    sub2.register_forward_hook.return_value = MagicMock()
    return model


def _make_mock_autoconfig(vocab_size=32000, model_type="llama"):
    cfg = MagicMock(name="AutoConfig_instance")
    cfg.vocab_size = vocab_size
    cfg.model_type = model_type
    return cfg


# ---------------------------------------------------------------------------
# fx_trace.py
# ---------------------------------------------------------------------------

class TestRunFxTrace:
    def test_returns_false_when_torch_unavailable(self):
        """Should return False (not raise) when torch is not installed."""
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            # Force re-import to pick up the mocked modules
            if "modelsig.torch.fx_trace" in sys.modules:
                del sys.modules["modelsig.torch.fx_trace"]
            from modelsig.torch.fx_trace import run_fx_trace
            result = run_fx_trace("test/model", None)
            assert result is False

    def test_returns_false_on_exception(self):
        """Should return False when symbolic_trace raises."""
        mock_torch = _make_mock_torch()
        mock_cfg = _make_mock_autoconfig()
        mock_model = _make_mock_model()

        mock_automodelcls = MagicMock()
        mock_automodelcls.from_config.return_value = mock_model
        mock_autoconfig_cls = MagicMock()
        mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

        mock_transformers = MagicMock(name="transformers")
        mock_transformers.AutoConfig = mock_autoconfig_cls
        mock_transformers.AutoModelForCausalLM = mock_automodelcls

        mock_fx = MagicMock(name="torch.fx")
        mock_fx.symbolic_trace.side_effect = RuntimeError("trace failed")
        mock_torch.fx = mock_fx

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.fx": mock_fx,
            "transformers": mock_transformers,
        }):
            if "modelsig.torch.fx_trace" in sys.modules:
                del sys.modules["modelsig.torch.fx_trace"]
            from modelsig.torch.fx_trace import run_fx_trace
            result = run_fx_trace("test/model", None)
            assert result is False

    def test_returns_true_when_trace_succeeds(self):
        """Should return True when symbolic_trace succeeds."""
        mock_torch = _make_mock_torch()
        mock_cfg = _make_mock_autoconfig()
        mock_model = _make_mock_model()

        mock_automodelcls = MagicMock()
        mock_automodelcls.from_config.return_value = mock_model
        mock_autoconfig_cls = MagicMock()
        mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

        mock_transformers = MagicMock(name="transformers")
        mock_transformers.AutoConfig = mock_autoconfig_cls
        mock_transformers.AutoModelForCausalLM = mock_automodelcls

        mock_fx = MagicMock(name="torch.fx")
        mock_fx.symbolic_trace.return_value = MagicMock(name="traced_graph")
        mock_torch.fx = mock_fx

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.fx": mock_fx,
            "transformers": mock_transformers,
        }):
            if "modelsig.torch.fx_trace" in sys.modules:
                del sys.modules["modelsig.torch.fx_trace"]
            from modelsig.torch.fx_trace import run_fx_trace
            result = run_fx_trace("test/model", None)
            assert result is True

    def test_uses_local_path_when_provided(self):
        """When local_path is provided, it should be used instead of model_id."""
        mock_torch = _make_mock_torch()
        mock_cfg = _make_mock_autoconfig()
        mock_model = _make_mock_model()

        mock_automodelcls = MagicMock()
        mock_automodelcls.from_config.return_value = mock_model
        mock_autoconfig_cls = MagicMock()
        mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

        mock_transformers = MagicMock(name="transformers")
        mock_transformers.AutoConfig = mock_autoconfig_cls
        mock_transformers.AutoModelForCausalLM = mock_automodelcls

        mock_fx = MagicMock(name="torch.fx")
        mock_fx.symbolic_trace.return_value = MagicMock()
        mock_torch.fx = mock_fx

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.fx": mock_fx,
            "transformers": mock_transformers,
        }):
            if "modelsig.torch.fx_trace" in sys.modules:
                del sys.modules["modelsig.torch.fx_trace"]
            from modelsig.torch.fx_trace import run_fx_trace
            run_fx_trace("remote/model", "/local/path/to/model")
            mock_autoconfig_cls.from_pretrained.assert_called_with(
                "/local/path/to/model", trust_remote_code=True
            )


# ---------------------------------------------------------------------------
# hooks.py
# ---------------------------------------------------------------------------

class TestRunHookCapture:
    def test_returns_empty_when_torch_unavailable(self):
        """Should return {} (not raise) when torch is not installed."""
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            if "modelsig.torch.hooks" in sys.modules:
                del sys.modules["modelsig.torch.hooks"]
            from modelsig.torch.hooks import run_hook_capture
            result = run_hook_capture("test/model", None)
            assert result == {}

    def test_returns_empty_on_exception(self):
        """Should return {} when model loading fails."""
        mock_transformers = MagicMock(name="transformers")
        mock_transformers.AutoConfig.from_pretrained.side_effect = RuntimeError("load failed")

        with patch.dict("sys.modules", {
            "torch": _make_mock_torch(),
            "transformers": mock_transformers,
        }):
            if "modelsig.torch.hooks" in sys.modules:
                del sys.modules["modelsig.torch.hooks"]
            from modelsig.torch.hooks import run_hook_capture
            result = run_hook_capture("test/model", None)
            assert result == {}

    def test_hooks_registered_and_removed(self):
        """Each module should have a hook registered and then removed."""
        mock_torch = _make_mock_torch()
        mock_cfg = _make_mock_autoconfig()
        mock_model = _make_mock_model()

        sub1 = MagicMock(name="layer0")
        sub2 = MagicMock(name="layer1")
        hook1 = MagicMock(name="hook_handle_0")
        hook2 = MagicMock(name="hook_handle_1")
        sub1.register_forward_hook.return_value = hook1
        sub2.register_forward_hook.return_value = hook2
        mock_model.named_modules.return_value = [("layer0", sub1), ("layer1", sub2)]

        mock_automodelcls = MagicMock()
        mock_automodelcls.from_config.return_value = mock_model
        mock_autoconfig_cls = MagicMock()
        mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

        mock_transformers = MagicMock(name="transformers")
        mock_transformers.AutoConfig = mock_autoconfig_cls
        mock_transformers.AutoModelForCausalLM = mock_automodelcls

        # Simulate forward pass calling the hooks
        def fake_call(dummy):
            # Trigger hook callbacks for each module
            return MagicMock()
        mock_model.side_effect = fake_call

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }):
            if "modelsig.torch.hooks" in sys.modules:
                del sys.modules["modelsig.torch.hooks"]
            from modelsig.torch.hooks import run_hook_capture
            run_hook_capture("test/model", None)

        # Verify hooks were registered
        sub1.register_forward_hook.assert_called_once()
        sub2.register_forward_hook.assert_called_once()
        # Verify hooks were removed
        hook1.remove.assert_called_once()
        hook2.remove.assert_called_once()

    def test_returns_dict(self):
        """Return value should always be a dict (possibly empty)."""
        mock_torch = _make_mock_torch()
        mock_cfg = _make_mock_autoconfig()
        mock_model = _make_mock_model()

        mock_automodelcls = MagicMock()
        mock_automodelcls.from_config.return_value = mock_model
        mock_autoconfig_cls = MagicMock()
        mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

        mock_transformers = MagicMock(name="transformers")
        mock_transformers.AutoConfig = mock_autoconfig_cls
        mock_transformers.AutoModelForCausalLM = mock_automodelcls

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }):
            if "modelsig.torch.hooks" in sys.modules:
                del sys.modules["modelsig.torch.hooks"]
            from modelsig.torch.hooks import run_hook_capture
            result = run_hook_capture("test/model", None)
            assert isinstance(result, dict)

    def test_uses_local_path_when_provided(self):
        """When local_path is provided it should be passed to AutoConfig."""
        mock_torch = _make_mock_torch()
        mock_cfg = _make_mock_autoconfig()
        mock_model = _make_mock_model()

        mock_automodelcls = MagicMock()
        mock_automodelcls.from_config.return_value = mock_model
        mock_autoconfig_cls = MagicMock()
        mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

        mock_transformers = MagicMock(name="transformers")
        mock_transformers.AutoConfig = mock_autoconfig_cls
        mock_transformers.AutoModelForCausalLM = mock_automodelcls

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }):
            if "modelsig.torch.hooks" in sys.modules:
                del sys.modules["modelsig.torch.hooks"]
            from modelsig.torch.hooks import run_hook_capture
            run_hook_capture("remote/model", "/local/path")
            mock_autoconfig_cls.from_pretrained.assert_called_with(
                "/local/path", trust_remote_code=True
            )
