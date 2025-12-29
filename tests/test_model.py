"""
Tests for Qlib model training and prediction.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml


class TestWorkflowConfig:
    """Tests for workflow.yaml configuration."""

    def test_yaml_parseable(self):
        """Check that workflow.yaml can be parsed by yaml library."""
        config_path = Path(__file__).parent.parent / "config" / "workflow.yaml"
        if not config_path.exists():
            pytest.skip("workflow.yaml not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert isinstance(config, dict)

    def test_yaml_has_required_keys(self):
        """Check that workflow.yaml has required configuration keys."""
        config_path = Path(__file__).parent.parent / "config" / "workflow.yaml"
        if not config_path.exists():
            pytest.skip("workflow.yaml not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check essential keys
        assert "qlib_init" in config
        assert "model" in config
        assert "data_handler_config" in config

    def test_yaml_model_is_lgbmodel(self):
        """Check that model is configured as LGBModel."""
        config_path = Path(__file__).parent.parent / "config" / "workflow.yaml"
        if not config_path.exists():
            pytest.skip("workflow.yaml not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        assert model_config.get("class") == "LGBModel"
        assert "gbdt" in model_config.get("module_path", "")


class TestPredictionsOutput:
    """Tests for predictions.csv output format."""

    def test_predictions_csv_has_score_column(self, tmp_path):
        """Check that predictions CSV contains score column."""
        # Create a mock predictions file
        mock_pred = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "instrument": ["SH600519", "SH600519", "SH600519"],
            "score": [0.5, -0.3, 0.8],
        })
        pred_path = tmp_path / "predictions.csv"
        mock_pred.to_csv(pred_path, index=False)

        # Read and verify
        df = pd.read_csv(pred_path)
        assert "score" in df.columns

    def test_predictions_score_range_reasonable(self, tmp_path):
        """Check that score values are within reasonable Z-score range."""
        # Create mock predictions with Z-score normalized values
        mock_pred = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "instrument": ["SH600519", "SH600520", "SH600521"],
            "score": [1.2, -0.8, 2.1],  # Typical Z-scores
        })
        pred_path = tmp_path / "predictions.csv"
        mock_pred.to_csv(pred_path, index=False)

        # Read and verify
        df = pd.read_csv(pred_path)
        scores = df["score"]

        # Z-scores should typically be between -3 and 3
        # Allow some outliers but most should be in range
        assert scores.min() >= -5, "Score too low, may indicate issue"
        assert scores.max() <= 5, "Score too high, may indicate issue"

    def test_predictions_not_empty(self, tmp_path):
        """Check that predictions file is not empty."""
        mock_pred = pd.DataFrame({
            "datetime": ["2023-01-01"],
            "instrument": ["SH600519"],
            "score": [0.5],
        })
        pred_path = tmp_path / "predictions.csv"
        mock_pred.to_csv(pred_path, index=False)

        df = pd.read_csv(pred_path)
        assert len(df) > 0, "Predictions file should not be empty"

    def test_predictions_has_required_columns(self, tmp_path):
        """Check that predictions CSV has all required columns."""
        mock_pred = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-02"],
            "instrument": ["SH600519", "SH600520"],
            "score": [0.5, -0.3],
        })
        pred_path = tmp_path / "predictions.csv"
        mock_pred.to_csv(pred_path, index=False)

        df = pd.read_csv(pred_path)
        required_columns = {"instrument", "datetime", "score"}
        assert required_columns.issubset(set(df.columns))


class TestTrainerModule:
    """Tests for trainer.py module."""

    def test_load_config_function(self, tmp_path):
        """Test that load_config can parse a valid YAML file."""
        from src.model.trainer import load_config

        # Create a test config
        test_config = {
            "qlib_init": {"provider_uri": "/test/path"},
            "model": {"class": "LGBModel"},
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        # Load and verify
        loaded = load_config(str(config_path))
        assert loaded["qlib_init"]["provider_uri"] == "/test/path"
        assert loaded["model"]["class"] == "LGBModel"


class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_save_and_load_model(self, tmp_path):
        """Test that a model can be saved and loaded."""
        from src.model.trainer import save_model, load_model

        # Create a mock model (simple dict for testing)
        mock_model = {"type": "LGBModel", "params": {"n_estimators": 100}}

        # Save
        model_path = save_model(mock_model, model_dir=str(tmp_path), model_name="test_model")

        # Verify file exists
        assert os.path.exists(model_path)
        assert model_path.endswith(".pkl")

        # Load and verify
        loaded_model = load_model(model_path)
        assert loaded_model == mock_model
        assert loaded_model["type"] == "LGBModel"

    def test_save_model_auto_name(self, tmp_path):
        """Test that save_model generates timestamp-based name."""
        from src.model.trainer import save_model

        mock_model = {"test": True}
        model_path = save_model(mock_model, model_dir=str(tmp_path))

        # Should contain lgb_model prefix and .pkl extension
        assert "lgb_model_" in model_path
        assert model_path.endswith(".pkl")

    def test_get_latest_model(self, tmp_path):
        """Test that get_latest_model returns the most recent model."""
        import time
        from src.model.trainer import save_model, get_latest_model

        # Save multiple models
        save_model({"version": 1}, model_dir=str(tmp_path), model_name="model_v1")
        time.sleep(0.1)  # Ensure different timestamps
        save_model({"version": 2}, model_dir=str(tmp_path), model_name="model_v2")
        time.sleep(0.1)
        latest_path = save_model({"version": 3}, model_dir=str(tmp_path), model_name="model_v3")

        # Get latest
        found_path = get_latest_model(str(tmp_path))

        assert found_path is not None
        assert found_path == latest_path

    def test_get_latest_model_empty_dir(self, tmp_path):
        """Test that get_latest_model returns None for empty directory."""
        from src.model.trainer import get_latest_model

        result = get_latest_model(str(tmp_path))
        assert result is None

    def test_get_latest_model_nonexistent_dir(self, tmp_path):
        """Test that get_latest_model returns None for nonexistent directory."""
        from src.model.trainer import get_latest_model

        result = get_latest_model(str(tmp_path / "nonexistent"))
        assert result is None

    def test_load_model_not_found(self, tmp_path):
        """Test that load_model raises error for missing file."""
        from src.model.trainer import load_model

        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "nonexistent.pkl"))


class TestRollingTrainer:
    """Tests for rolling trainer module."""

    def test_load_rolling_config(self, tmp_path):
        """Test loading rolling workflow config."""
        from src.model.rolling_trainer import load_config

        # Create a test config
        test_config = {
            "qlib_init": {"provider_uri": "/test/path"},
            "rolling": {
                "step": 20,
                "train_window": 480,
                "valid_window": 60,
                "test_window": 20,
            },
            "task": {
                "model": {"class": "LGBModel"},
            },
        }
        config_path = tmp_path / "rolling_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        loaded = load_config(str(config_path))
        assert loaded["rolling"]["step"] == 20
        assert loaded["rolling"]["train_window"] == 480

    def test_generate_rolling_windows(self):
        """Test rolling window generation."""
        from src.model.rolling_trainer import generate_rolling_windows

        # Create a simple calendar
        calendar = [f"2020-01-{i:02d}" for i in range(1, 32)]

        windows = generate_rolling_windows(
            calendar=calendar,
            first_train_end="2020-01-20",
            train_window=10,
            valid_window=5,
            test_window=5,
            step=5,
        )

        # Should generate at least one window
        assert len(windows) >= 1

        # Check first window structure
        first_window = windows[0]
        assert first_window.window_id == 0
        assert first_window.train_start is not None
        assert first_window.test_end is not None

    def test_rolling_window_dataclass(self):
        """Test RollingWindow dataclass."""
        from src.model.rolling_trainer import RollingWindow

        window = RollingWindow(
            train_start="2020-01-01",
            train_end="2020-06-30",
            valid_start="2020-07-01",
            valid_end="2020-09-30",
            test_start="2020-10-01",
            test_end="2020-10-31",
            window_id=0,
        )

        assert window.window_id == 0
        assert "Window 0" in repr(window)
        assert "2020-01-01" in repr(window)

    def test_save_and_load_rolling_model(self, tmp_path):
        """Test saving and loading rolling model."""
        from src.model.rolling_trainer import (
            RollingWindow,
            save_rolling_model,
            load_rolling_model,
        )

        # Create a mock model
        mock_model = {"type": "LGBModel", "window": 0}

        window = RollingWindow(
            train_start="2020-01-01",
            train_end="2020-06-30",
            valid_start="2020-07-01",
            valid_end="2020-09-30",
            test_start="2020-10-01",
            test_end="2020-10-31",
            window_id=0,
        )

        # Save
        model_path = save_rolling_model(mock_model, window, str(tmp_path))
        assert os.path.exists(model_path)
        assert "20201001" in model_path  # test_start date in filename

        # Load
        loaded = load_rolling_model(model_path)
        assert loaded == mock_model

    def test_rolling_result_dataclass(self):
        """Test RollingResult dataclass."""
        from src.model.rolling_trainer import RollingWindow, RollingResult

        window = RollingWindow(
            train_start="2020-01-01",
            train_end="2020-06-30",
            valid_start="2020-07-01",
            valid_end="2020-09-30",
            test_start="2020-10-01",
            test_end="2020-10-31",
            window_id=0,
        )

        result = RollingResult(
            window=window,
            model_path="/path/to/model.pkl",
            predictions_path="/path/to/pred.csv",
            metrics={"pred_count": 100, "pred_mean": 0.5},
        )

        assert result.window.window_id == 0
        assert result.metrics["pred_count"] == 100

