"""Load CICIoMT and run the configured statistical classification models."""

from pathlib import Path

import pandas as pd
import yaml

from src.machineLearning.statistical import run_statistical_models


def main() -> pd.DataFrame:
    config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    with config_path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    dataset_config = config["dataset"]
    dataset_path = (config_path.parent / dataset_config["path"]).resolve()
    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = pd.read_csv(dataset_path, compression=dataset_config["compression"])
    return run_statistical_models(dataset, config)


if __name__ == "__main__":
    results = main()
