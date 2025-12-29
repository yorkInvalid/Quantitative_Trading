"""
Qlib model trainer and predictor.

Initializes Qlib, runs the workflow from config/workflow.yaml,
saves the trained model, and outputs predictions to CSV.

Supports model persistence (save/load) for production use.
"""

import importlib
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
import yaml

import qlib
from qlib.config import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

# Default paths
DEFAULT_CONFIG_PATH = "/app/config/workflow.yaml"
DEFAULT_PROVIDER_URI = "/app/data/qlib_bin"
DEFAULT_OUTPUT_PATH = "/app/data/predictions.csv"
DEFAULT_MODEL_DIR = "/app/data/models/trained"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_qlib(provider_uri: str = DEFAULT_PROVIDER_URI, region: str = "cn") -> None:
    """Initialize Qlib with the specified data provider."""
    qlib.init(provider_uri=provider_uri, region=REG_CN if region == "cn" else region)
    print(f"Qlib initialized with provider_uri: {provider_uri}")


def build_model_from_config(config: dict) -> Any:
    """
    Build model instance from configuration.
    
    Args:
        config: Full configuration dict containing 'model' key.
        
    Returns:
        Instantiated model object.
    """
    model_config = config.get("model", {})
    model_class = model_config.get("class", "LGBModel")
    model_module = model_config.get("module_path", "qlib.contrib.model.gbdt")
    model_kwargs = model_config.get("kwargs", {})

    module = importlib.import_module(model_module)
    ModelClass = getattr(module, model_class)
    return ModelClass(**model_kwargs)


def build_dataset_from_config(config: dict) -> Any:
    """
    Build dataset instance from configuration.
    
    Args:
        config: Full configuration dict containing 'dataset' key.
        
    Returns:
        Instantiated dataset object.
    """
    dataset_config = config.get("dataset", {})
    dataset_class = dataset_config.get("class", "DatasetH")
    dataset_module = dataset_config.get("module_path", "qlib.data.dataset")

    module = importlib.import_module(dataset_module)
    DatasetClass = getattr(module, dataset_class)

    # Build handler
    handler_config = dataset_config.get("kwargs", {}).get("handler", {})
    handler_class = handler_config.get("class", "DataHandlerLP")
    handler_module = handler_config.get("module_path", "qlib.data.dataset.handler")
    handler_kwargs = handler_config.get("kwargs", config.get("data_handler_config", {}))

    module = importlib.import_module(handler_module)
    HandlerClass = getattr(module, handler_class)
    handler = HandlerClass(**handler_kwargs)

    segments = dataset_config.get("kwargs", {}).get("segments", {})
    return DatasetClass(handler=handler, segments=segments)


def save_model(
    model: Any,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_name: Optional[str] = None,
) -> str:
    """
    Save trained model to disk using pickle.
    
    Args:
        model: Trained model object.
        model_dir: Directory to save the model.
        model_name: Optional custom name. Defaults to timestamp-based name.
        
    Returns:
        Path to the saved model file.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lgb_model_{timestamp}.pkl"
    
    if not model_name.endswith(".pkl"):
        model_name += ".pkl"
    
    model_path = os.path.join(model_dir, model_name)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    return model_path


def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file.
        
    Returns:
        Loaded model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {model_path}")
    return model


def get_latest_model(model_dir: str = DEFAULT_MODEL_DIR) -> Optional[str]:
    """
    Get the path to the most recently saved model.
    
    Args:
        model_dir: Directory containing saved models.
        
    Returns:
        Path to the latest model file, or None if no models exist.
    """
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        return None
    
    model_files = list(model_dir_path.glob("*.pkl"))
    if not model_files:
        return None
    
    # Sort by modification time, get the latest
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def train_model(
    config: dict,
    experiment_name: str = "quant_lgb",
    save_to_disk: bool = True,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> Tuple[Any, str]:
    """
    Train model using Qlib workflow.

    Args:
        config: Configuration dictionary.
        experiment_name: Name for the Qlib experiment.
        save_to_disk: Whether to save the trained model.
        model_dir: Directory to save the model.
        
    Returns:
        Tuple of (trained_model, recorder_id).
    """
    with R.start(experiment_name=experiment_name):
        R.log_params(**{"config": str(config)})

        # Build model and dataset
        model = build_model_from_config(config)
        dataset = build_dataset_from_config(config)

        # Train
        model.fit(dataset)

        # Record signals
        recorder = R.get_recorder()
        sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
        sr.generate()

        recorder_id = recorder.id
        print(f"Training completed. Recorder ID: {recorder_id}")

    # Save model to disk
    if save_to_disk:
        save_model(model, model_dir=model_dir)

    return model, recorder_id


def predict_and_save(
    config: dict,
    output_path: str = DEFAULT_OUTPUT_PATH,
    model: Optional[Any] = None,
    model_path: Optional[str] = None,
    segment: str = "test",
) -> pd.DataFrame:
    """
    Generate predictions using a trained model.

    Can use either a provided model object, load from a saved model file,
    or train a new model if neither is provided.
    
    Args:
        config: Configuration dictionary.
        output_path: Path to save predictions CSV.
        model: Optional pre-trained model object.
        model_path: Optional path to load model from disk.
        segment: Dataset segment to predict on ("train", "valid", "test").

    Returns:
        DataFrame with predictions (instrument, datetime, score).
    """
    # Load or build model
    if model is not None:
        print("Using provided model object")
    elif model_path is not None:
        model = load_model(model_path)
    else:
        # Try to load the latest saved model
        latest_model_path = get_latest_model()
        if latest_model_path:
            print(f"Loading latest model: {latest_model_path}")
            model = load_model(latest_model_path)
        else:
            # No saved model, need to train
            print("No saved model found. Training new model...")
            model = build_model_from_config(config)
            dataset = build_dataset_from_config(config)
            model.fit(dataset)

    # Build dataset for prediction
    dataset = build_dataset_from_config(config)

    # Predict on specified segment
    pred = model.predict(dataset, segment=segment)

    # Format output
    if isinstance(pred, pd.Series):
        pred_df = pred.reset_index()
        pred_df.columns = ["datetime", "instrument", "score"]
    else:
        pred_df = pred.reset_index()
        if len(pred_df.columns) == 3:
            pred_df.columns = ["datetime", "instrument", "score"]
        else:
            pred_df.columns = list(pred_df.columns[:-1]) + ["score"]

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Shape: {pred_df.shape}, Columns: {list(pred_df.columns)}")

    return pred_df


def run_workflow(
    config_path: str = DEFAULT_CONFIG_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    save_model_to_disk: bool = True,
    model_dir: str = DEFAULT_MODEL_DIR,
    use_saved_model: bool = False,
) -> pd.DataFrame:
    """
    Full workflow: init -> train -> predict -> save.
    
    Args:
        config_path: Path to workflow.yaml configuration.
        output_path: Path to save predictions CSV.
        save_model_to_disk: Whether to save the trained model.
        model_dir: Directory to save/load models.
        use_saved_model: If True, try to load existing model instead of training.
        
    Returns:
        DataFrame with predictions.
    """
    # Load config
    config = load_config(config_path)

    # Initialize Qlib
    qlib_config = config.get("qlib_init", {})
    provider_uri = qlib_config.get("provider_uri", DEFAULT_PROVIDER_URI)
    region = qlib_config.get("region", "cn")
    init_qlib(provider_uri=provider_uri, region=region)

    # Train or load model
    model = None
    if use_saved_model:
        latest_model_path = get_latest_model(model_dir)
        if latest_model_path:
            model = load_model(latest_model_path)
        else:
            print("No saved model found. Training new model...")
    
    if model is None:
        model, _ = train_model(
            config,
            save_to_disk=save_model_to_disk,
            model_dir=model_dir,
        )

    # Predict and save
    pred_df = predict_and_save(config, output_path=output_path, model=model)

    return pred_df


def predict_only(
    config_path: str = DEFAULT_CONFIG_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    model_path: Optional[str] = None,
    segment: str = "test",
) -> pd.DataFrame:
    """
    Prediction-only workflow using a saved model (no training).
    
    Args:
        config_path: Path to workflow.yaml configuration.
        output_path: Path to save predictions CSV.
        model_path: Path to saved model. If None, uses latest model.
        segment: Dataset segment to predict on.
        
    Returns:
        DataFrame with predictions.
    """
    # Load config
    config = load_config(config_path)

    # Initialize Qlib
    qlib_config = config.get("qlib_init", {})
    provider_uri = qlib_config.get("provider_uri", DEFAULT_PROVIDER_URI)
    region = qlib_config.get("region", "cn")
    init_qlib(provider_uri=provider_uri, region=region)

    # Load model
    if model_path is None:
        model_path = get_latest_model()
        if model_path is None:
            raise FileNotFoundError(
                "No saved model found. Run training first or specify model_path."
            )
    
    model = load_model(model_path)

    # Predict and save
    pred_df = predict_and_save(
        config,
        output_path=output_path,
        model=model,
        segment=segment,
    )

    return pred_df


if __name__ == "__main__":
    run_workflow()

