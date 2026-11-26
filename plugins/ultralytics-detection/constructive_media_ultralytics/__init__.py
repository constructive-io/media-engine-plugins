"""Ultralytics YOLO detectors for Constructive Media.

Fills the engine's `detection` slot under the name `ultralytics`.

Ultralytics' code and the models it publishes are licensed AGPL-3.0, so this
adapter is too, and that is why it lives here and not in the engine. Read the
license before you use it for anything you serve to other people.

    Copyright (C) 2026 Dan Lynch <pyramation@gmail.com>
    Copyright (C) 2026 Constructive <developers@constructive.io>

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU Affero General Public License, version 3, as
    published by the Free Software Foundation. It is distributed WITHOUT ANY
    WARRANTY; see the LICENSE file beside this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from constructive_media.adapters import Adapter, lazy
from constructive_media.detection import Detection
from constructive_media.settings import Settings

__version__ = "0.1.0"

LICENSE = "AGPL-3.0-only"

DEFAULT_MODEL = "yolo11n"
BATCH = 16


def adapters() -> list[Adapter]:
    """What this plugin gives the engine."""
    return [
        Adapter(
            slot="detection",
            name="ultralytics",
            factory=lazy("constructive_media_ultralytics:UltralyticsDetector"),
            license=LICENSE,
            summary="Objects and people with Ultralytics YOLO models",
        )
    ]


class UltralyticsDetector:
    """Any detection model Ultralytics can load, by name or by path.

    The model is `CONSTRUCTIVE_MEDIA_MODEL_DETECTION`: a name Ultralytics
    publishes (`yolo11n`, `yolo11s`, ...), which it downloads on first use into
    the engine's models folder, or the path of a model file of your own.
    """

    name = "ultralytics"

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or Settings.from_env()
        self.model = model or settings.model("detection") or DEFAULT_MODEL
        self.device = device or settings.device
        self.folder = settings.models_dir / "ultralytics"
        self._network: Any = None

    @property
    def weights(self) -> Path:
        """Where the model's file is, or is to be downloaded to."""
        given = Path(self.model).expanduser()
        if given.suffix == "":
            given = given.with_suffix(".pt")
        if given.is_file() or given.parent != Path(".") or given.suffix in (".yaml", ".yml"):
            return given
        return self.folder / given.name

    def _load(self) -> Any:
        if self._network is None:
            import torch
            from ultralytics import YOLO

            weights = self.weights
            if weights.parent == self.folder:
                self.folder.mkdir(parents=True, exist_ok=True)
            if self.device == "auto":
                self.device = ("cuda" if torch.cuda.is_available()
                               else "mps" if torch.backends.mps.is_available() else "cpu")
            self._network = YOLO(str(weights), task="detect")
        return self._network

    def detect(self, images: Sequence[Path], confidence: float = 0.35) -> list[list[Detection]]:
        network = self._load()
        found: list[list[Detection]] = []
        for start in range(0, len(images), BATCH):
            batch = [str(path) for path in images[start:start + BATCH]]
            for result in network.predict(batch, conf=confidence, device=self.device, verbose=False):
                height, width = result.orig_shape
                boxes = result.boxes
                found.append([
                    Detection.from_corners(str(result.names[int(kind)]), score, *box, width, height)
                    for box, kind, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
                ])
        return found
