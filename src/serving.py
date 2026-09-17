"""Load the selected pipeline from a checksum-verified, portable model artifact."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd
import skops.io as sio

from src.config import ROOT, feature_names, load_config
from src.interface import validate_features

TRUSTED_TYPES = {"numpy.dtype", "src.preprocessing.FrameValidator", "sklearn.tree._tree.Tree"}
MODEL_DIR = ROOT / "artifacts" / "serving"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def ensure_model(directory: Path = MODEL_DIR) -> dict:
    manifest = json.loads((ROOT / "configs" / "model_release.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.skops"
    if not path.exists() or sha256(path) != manifest["sha256"]:
        url = manifest.get("url", "")
        if not url.startswith("https://github.com/MichaelMartiradonna/ames-housing-mlops/releases/download/"):
            raise ValueError("Model artifact is unavailable. Run python -m src.export_model after training.")
        temporary = directory / f"download-{uuid4().hex}.tmp"
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                size = 0
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 100_000_000:
                            raise ValueError("Model download exceeds the expected size.")
                        output.write(chunk)
            if sha256(temporary) != manifest["sha256"]:
                raise ValueError("Model checksum does not match the release manifest.")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def load_serving_model(directory: Path = MODEL_DIR):
    metadata = ensure_model(directory)
    path = directory / "model.skops"
    untrusted = set(sio.get_untrusted_types(file=path))
    if not untrusted <= TRUSTED_TYPES:
        raise ValueError("Model contains unapproved object types.")
    return sio.load(path, trusted=list(TRUSTED_TYPES)), metadata


def predict(model, metadata: dict, features: dict) -> float:
    review = validate_features(features, metadata["categories"])
    if not review.ready:
        raise ValueError("Complete and validate all home details before estimating.")
    frame = pd.DataFrame([review.features], columns=feature_names(load_config()))
    prediction = np.asarray(model.predict(frame))
    if prediction.shape != (1,) or not np.isfinite(prediction).all() or prediction[0] <= 0:
        raise ValueError("The model did not return a valid price.")
    return float(prediction[0])
