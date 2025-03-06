import os
import yaml
from ultralytics import YOLO
import shutil
import datetime

# ------------------------------------------------------------------------------
#  Point to your real data directory (where images, metadata, training folder are).
# ------------------------------------------------------------------------------
DATA_BASE_DIR = "/home/user/WatchMyBirds-Data"
IMAGE_UNPROCESSED_DIR = os.path.join(DATA_BASE_DIR, "images", "unprocessed")
METADATA_FILE = os.path.join(DATA_BASE_DIR, "metadata", "dataset.csv")

TRAIN_FOLDER = os.path.join(DATA_BASE_DIR, "training")
CONFIGS_FOLDER = os.path.join(TRAIN_FOLDER, "configs")  # train_config.yaml
DATA_FOLDER = os.path.join(TRAIN_FOLDER, "data")  # data.yaml

# ------------------------------------------------------------------------------
# Append current date/time to the run name
# ------------------------------------------------------------------------------
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

model_to_train = "yolov8n.pt"

base_model_name = os.path.splitext(os.path.basename(model_to_train))[0]
RUNS_FOLDER = os.path.join(TRAIN_FOLDER, f"runs_{base_model_name}")

run_name = f"{base_model_name}_watch_my_birds-{current_time}"


def load_config(config_path="train_config.yaml"):
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    # 1) Load training config
    config_file = "train_config.yaml"
    config_path = os.path.join(CONFIGS_FOLDER, config_file)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"train_config.yaml not found at {config_path}")

    cfg = load_config(config_path)
    train_cfg = cfg["train"]

    # 2) Initialize YOLO model
    model = YOLO(model_to_train)

    # 3) Load data.yaml
    data_file = os.path.join(DATA_FOLDER, "data.yaml")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"data.yaml not found at {data_file}")

    with open(data_file, "r") as f:
        data_yaml = yaml.safe_load(f)  # ✅ Load data.yaml here

    print(f"✅ data.yaml is correctly set to: {data_file}")

    train_params = {
        "data": str(data_file),  # Points to data.yaml
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch"],
        "imgsz": train_cfg["imgsz"],
        "lr0": train_cfg.get("lr0", 0.0055),  # Initial learning rate
        "patience": train_cfg.get("patience", 10),
        "optimizer": train_cfg.get("optimizer", "SGD"),
        "augment": train_cfg.get("augment", False),
        "cache": train_cfg.get("cache", False),
        "single_cls": train_cfg.get("single_cls", False),
        "plots": train_cfg.get("plots", False),
        "multi_scale": train_cfg.get("multi_scale", False),
        "dropout": train_cfg.get("dropout", 0.0),
        "workers": train_cfg.get("workers", 8),
        "project": RUNS_FOLDER,
        "name": run_name,
        "save": True,
        "seed": 42,
        "verbose": True,
        "auto_augment": "randaugment",
        "bgr": 0.0,
        "fliplr": 0.5,
        "flipud": 0.5,
        "perspective": 0.0,
        "shear": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "hsv_v": 0.4,
        "hsv_s": 0.0,
        "hsv_h": 0.015,
        "copy_paste": 0.2,
        "mosaic": 0.2,
        "mixup": 0.0,
        "copy_paste_mode": "flip",
        "erasing": 0.4,
        "crop_fraction": 0.5,
    }

    # Only add the learning rate if optimizer is not set to "auto"
    if train_cfg["optimizer"] != "auto":
        train_params["lr0"] = train_cfg.get("lr0", 0.005)

    # 4) Train the YOLO model
    results = model.train(**train_params)

    print("✅ Training finished. Best model should be in "
          f"{os.path.join(RUNS_FOLDER, run_name, 'weights', 'best.pt')}")

    # --- Export the best model to ONNX format ---
    best_model_path = os.path.join(RUNS_FOLDER, run_name, "weights", "best.pt")
    print(f"Exporting best model from {best_model_path} to ONNX format.")
    # Create a new YOLO instance with the best weights
    best_model = YOLO(best_model_path)
    # Export to ONNX with the specified image size and opset version
    best_model.export(format="onnx", imgsz=[640, 640], opset=12)
    print("✅ Export complete.")

    # ------------------------------------------------------------------------------
    # Copy configuration files to a training_config folder (excluding images)
    # ------------------------------------------------------------------------------
    # Create destination folder inside the run folder:
    training_config_folder = os.path.join(RUNS_FOLDER, run_name, "training_config")
    os.makedirs(training_config_folder, exist_ok=True)

    # List of config files to copy: train_config.yaml from CONFIGS_FOLDER, data.yaml and dataset_split.json from DATA_FOLDER
    config_files = {
        "train_config.yaml": os.path.join(CONFIGS_FOLDER, "train_config.yaml"),
        "data.yaml": os.path.join(DATA_FOLDER, "data.yaml"),
        "dataset_split.json": os.path.join(DATA_FOLDER, "dataset_split.json")
    }

    for fname, src_path in config_files.items():
        if os.path.isfile(src_path):
            dst_path = os.path.join(training_config_folder, fname)
            shutil.copy(src_path, dst_path)
            print(f"✅ Copied {fname} to {training_config_folder}")
        else:
            print(f"⚠️ File {src_path} not found, skipping.")
