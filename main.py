"""Read the CICIoMT dataset using the YAML configuration."""

from pathlib import Path

import pandas as pd
import yaml


def main() -> pd.DataFrame:
    config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    with config_path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    dataset_config = config["dataset"]
    dataset_path = (config_path.parent / dataset_config["path"]).resolve()
    return pd.read_csv(dataset_path, compression=dataset_config["compression"])


if __name__ == "__main__":
    dataset = main()
