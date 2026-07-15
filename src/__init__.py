"""Classical document-orientation preparation and modeling toolkit.

The package covers dataset inventory and privacy controls plus the rotation-zone
and exact-angle pipeline. It intentionally excludes OCR and neural models.
Heavy optional modules are not imported at package import time.
"""

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "angle_estimation",
    "config",
    "dataset_discovery",
    "dataset_validation",
    "duplicate_detection",
    "file_inventory",
    "orientation_features",
    "page_preparation",
    "privacy",
    "rotation_common",
    "rotation_dataset",
    "rotation_model",
    "stable_ids",
]
