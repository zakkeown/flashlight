"""
Serialization Tests

Tests for mlx_compat.save() and mlx_compat.load() functions.
"""

import pytest
import numpy as np
import tempfile
import os

import mlx_compat


class TestSaveTensor:
    """Tests for saving tensors."""

    def test_save_1d_tensor(self):
        """Test saving a 1D tensor."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(x, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_2d_tensor(self):
        """Test saving a 2D tensor."""
        x = mlx_compat.randn(10, 20)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(x, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_save_nd_tensor(self):
        """Test saving an N-dimensional tensor."""
        x = mlx_compat.randn(2, 3, 4, 5)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(x, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_save_integer_tensor(self):
        """Test saving an integer tensor."""
        x = mlx_compat.tensor([1, 2, 3, 4, 5], dtype=mlx_compat.int32)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(x, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestLoadTensor:
    """Tests for loading tensors."""

    def test_load_1d_tensor(self):
        """Test loading a 1D tensor."""
        original = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            np.testing.assert_allclose(
                np.array(loaded._mlx_array),
                np.array(original._mlx_array),
                rtol=1e-6,
            )
        finally:
            os.unlink(path)

    def test_load_2d_tensor(self):
        """Test loading a 2D tensor."""
        original = mlx_compat.randn(10, 20)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            np.testing.assert_allclose(
                np.array(loaded._mlx_array),
                np.array(original._mlx_array),
                rtol=1e-5,
            )
        finally:
            os.unlink(path)

    def test_load_preserves_shape(self):
        """Test that loading preserves tensor shape."""
        original = mlx_compat.randn(3, 4, 5)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert loaded.shape == original.shape
        finally:
            os.unlink(path)

    def test_load_preserves_dtype(self):
        """Test that loading preserves tensor dtype."""
        original = mlx_compat.tensor([1, 2, 3], dtype=mlx_compat.int32)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert loaded.dtype == original.dtype
        finally:
            os.unlink(path)


class TestSaveLoadDict:
    """Tests for saving and loading dictionaries (state dicts)."""

    def test_save_load_simple_dict(self):
        """Test saving and loading a simple dictionary."""
        original = {
            "weight": mlx_compat.randn(10, 5),
            "bias": mlx_compat.randn(5),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert set(loaded.keys()) == set(original.keys())
            np.testing.assert_allclose(
                np.array(loaded["weight"]._mlx_array),
                np.array(original["weight"]._mlx_array),
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                np.array(loaded["bias"]._mlx_array),
                np.array(original["bias"]._mlx_array),
                rtol=1e-5,
            )
        finally:
            os.unlink(path)

    def test_save_load_nested_dict(self):
        """Test saving and loading a nested dictionary."""
        original = {
            "layer1": {
                "weight": mlx_compat.randn(10, 5),
                "bias": mlx_compat.randn(5),
            },
            "layer2": {
                "weight": mlx_compat.randn(5, 2),
                "bias": mlx_compat.randn(2),
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert set(loaded.keys()) == set(original.keys())
            for layer in ["layer1", "layer2"]:
                for param in ["weight", "bias"]:
                    np.testing.assert_allclose(
                        np.array(loaded[layer][param]._mlx_array),
                        np.array(original[layer][param]._mlx_array),
                        rtol=1e-5,
                    )
        finally:
            os.unlink(path)

    def test_save_load_mixed_dict(self):
        """Test saving and loading a dict with tensors and scalars."""
        original = {
            "epoch": 10,
            "loss": 0.5,
            "weights": mlx_compat.randn(5, 5),
            "name": "my_model",
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert loaded["epoch"] == 10
            assert loaded["loss"] == 0.5
            assert loaded["name"] == "my_model"
            np.testing.assert_allclose(
                np.array(loaded["weights"]._mlx_array),
                np.array(original["weights"]._mlx_array),
                rtol=1e-5,
            )
        finally:
            os.unlink(path)


class TestSaveLoadList:
    """Tests for saving and loading lists."""

    def test_save_load_list_of_tensors(self):
        """Test saving and loading a list of tensors."""
        original = [
            mlx_compat.randn(5),
            mlx_compat.randn(10),
            mlx_compat.randn(3, 3),
        ]
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert len(loaded) == len(original)
            for i in range(len(original)):
                np.testing.assert_allclose(
                    np.array(loaded[i]._mlx_array),
                    np.array(original[i]._mlx_array),
                    rtol=1e-5,
                )
        finally:
            os.unlink(path)

    def test_save_load_tuple(self):
        """Test saving and loading a tuple of tensors."""
        original = (mlx_compat.randn(5), mlx_compat.randn(10))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert isinstance(loaded, tuple)
            assert len(loaded) == 2
        finally:
            os.unlink(path)


class TestModelStateDict:
    """Tests for saving and loading model state dicts."""

    def test_save_load_linear_state_dict(self):
        """Test saving and loading a Linear layer's state dict."""
        model = mlx_compat.nn.Linear(10, 5)
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(state_dict, path)
            loaded_state = mlx_compat.load(path)

            # Create new model and load state
            new_model = mlx_compat.nn.Linear(10, 5)
            new_model.load_state_dict(loaded_state)

            # Check parameters match
            for key in state_dict:
                orig_val = state_dict[key]
                new_val = new_model.state_dict()[key]
                # Handle both Tensor and raw array
                orig_arr = np.array(orig_val._mlx_array) if hasattr(orig_val, '_mlx_array') else np.array(orig_val)
                new_arr = np.array(new_val._mlx_array) if hasattr(new_val, '_mlx_array') else np.array(new_val)
                np.testing.assert_allclose(new_arr, orig_arr, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_save_load_sequential_state_dict(self):
        """Test saving and loading a Sequential model's state dict."""
        model = mlx_compat.nn.Sequential(
            mlx_compat.nn.Linear(10, 20),
            mlx_compat.nn.ReLU(),
            mlx_compat.nn.Linear(20, 5),
        )
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(state_dict, path)
            loaded_state = mlx_compat.load(path)

            new_model = mlx_compat.nn.Sequential(
                mlx_compat.nn.Linear(10, 20),
                mlx_compat.nn.ReLU(),
                mlx_compat.nn.Linear(20, 5),
            )
            new_model.load_state_dict(loaded_state)

            # Verify model works
            x = mlx_compat.randn(2, 10)
            output = new_model(x)
            assert output.shape == (2, 5)
        finally:
            os.unlink(path)


class TestCheckpointing:
    """Tests for checkpoint-style saving."""

    def test_save_load_checkpoint(self):
        """Test saving and loading a full checkpoint."""
        model = mlx_compat.nn.Linear(10, 5)
        optimizer_state = {"lr": 0.01, "step": 100}

        checkpoint = {
            "epoch": 50,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "best_loss": 0.123,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(checkpoint, path)
            loaded = mlx_compat.load(path)

            assert loaded["epoch"] == 50
            assert loaded["best_loss"] == 0.123
            assert loaded["optimizer_state"]["lr"] == 0.01
            assert loaded["optimizer_state"]["step"] == 100
            assert "model_state" in loaded
        finally:
            os.unlink(path)


class TestFileObjects:
    """Tests for saving/loading with file objects."""

    def test_save_to_file_object(self):
        """Test saving to an open file object."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            mlx_compat.save(x, f)
            path = f.name
        try:
            loaded = mlx_compat.load(path)
            np.testing.assert_allclose(
                np.array(loaded._mlx_array), [1.0, 2.0, 3.0], rtol=1e-5
            )
        finally:
            os.unlink(path)

    def test_load_from_file_object(self):
        """Test loading from an open file object."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(x, path)
            with open(path, "rb") as f:
                loaded = mlx_compat.load(f)
            np.testing.assert_allclose(
                np.array(loaded._mlx_array), [1.0, 2.0, 3.0], rtol=1e-5
            )
        finally:
            os.unlink(path)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_save_load_empty_dict(self):
        """Test saving and loading an empty dictionary."""
        original = {}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert loaded == {}
        finally:
            os.unlink(path)

    def test_save_load_scalar(self):
        """Test saving and loading a scalar value."""
        original = 42
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert loaded == 42
        finally:
            os.unlink(path)

    def test_save_load_string(self):
        """Test saving and loading a string."""
        original = "hello world"
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(original, path)
            loaded = mlx_compat.load(path)
            assert loaded == "hello world"
        finally:
            os.unlink(path)

    def test_requires_grad_preserved(self):
        """Test that requires_grad is preserved."""
        x = mlx_compat.randn(5, 5, requires_grad=True)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mlx_compat.save(x, path)
            loaded = mlx_compat.load(path)
            assert loaded.requires_grad == x.requires_grad
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
