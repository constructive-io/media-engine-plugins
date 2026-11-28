"""A voice taken out from over music for Constructive Media, through Demucs.

Fills the engine's `separation` slot under the name `demucs`. A recording
of somebody speaking or singing over music is written again as two: the
voice, and everything that was under it.

    media separate song.wav parts --adapter demucs

That writes `parts/voice.wav` and `parts/rest.wav`. Asked for every part,
it writes the voice and each thing the model keeps apart: `drums.wav`,
`bass.wav` and `other.wav` for the default model.

The code here is MIT, as the code of Demucs is. The weights are not: their
authors say they "are not covered by the MIT license, and are provided only
for scientific purposes", because some of the music they were trained on
could only be used for research. That is why this adapter lives here and
not in the engine, and why it says so wherever adapters are listed. The
README beside this package has what the authors say, in their words.

The model is `CONSTRUCTIVE_MEDIA_MODEL_SEPARATION` and what a recording is
pulled apart with is `CONSTRUCTIVE_MEDIA_EXTRA_SEPARATION`, a JSON object.
Both are read when this is the adapter chosen for separation, or when none
is and this one is asked for by name. The defaults are

    {"parts": "two", "shifts": 0, "overlap": 0.25}

`parts` is `two` for the voice and the rest, or `all` for the voice and
each of the model's other parts. `shifts` is how many times over the
recording is pulled apart, each from a start a little way off, and the
answers averaged: 0 does it once, and the same way every time. `overlap`
is how much each stretch the model works on shares with the next.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from constructive_media import gpu
from constructive_media.adapters import Adapter, lazy
from constructive_media.separation import REST, VOICE, SeparationError
from constructive_media.settings import Settings

__version__ = "0.1.0"

#: What the authors say of the weights, which is not a license: see the README.
LICENSE = "LicenseRef-Demucs-weights-research-only"
SLOT = "separation"
SETTING = "CONSTRUCTIVE_MEDIA_EXTRA_SEPARATION"

#: The model Demucs itself runs when it is not told which.
DEFAULT_MODEL = "htdemucs"
DEFAULTS: dict[str, Any] = {"parts": "two", "shifts": 0, "overlap": 0.25}
PARTS = ("two", "all")
#: Each shift is the whole recording pulled apart again.
MOST_SHIFTS = 10
#: What the models call the voice.
THEIRS = "vocals"
#: The models hear two channels, and this adapter hands them two.
CHANNELS = 2
DEVICES = ("mps", "cuda", "cpu")


def options(settings: Settings, chosen: bool = True) -> dict[str, Any]:
    """What a recording is pulled apart with: the defaults, and over them what the setting says."""
    written = settings.extra(SLOT) if chosen else None
    kept = dict(DEFAULTS)
    if written:
        try:
            given = json.loads(written)
        except ValueError as error:
            raise SeparationError(f"{SETTING} is not JSON: {error}") from None
        if not isinstance(given, dict):
            raise SeparationError(f'{SETTING} should be a JSON object, such as {{"parts": "all"}}')
        unknown = sorted(set(given) - set(DEFAULTS))
        if unknown:
            raise SeparationError(f"{SETTING} has {', '.join(unknown)}, which the demucs adapter does not know; "
                                  f"it knows: {', '.join(DEFAULTS)}")
        kept.update(given)
    parts, shifts, overlap = kept["parts"], kept["shifts"], kept["overlap"]
    if parts not in PARTS:
        raise SeparationError(f"the parts in {SETTING} is one of {', '.join(PARTS)}: not {parts!r}")
    if isinstance(shifts, bool) or not isinstance(shifts, int) or not 0 <= shifts <= MOST_SHIFTS:
        raise SeparationError(f"the shifts in {SETTING} is a whole number from 0 to {MOST_SHIFTS}: not {shifts!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or not 0 <= overlap < 1:
        raise SeparationError(f"the overlap in {SETTING} is a number from 0 to less than 1: not {overlap!r}")
    return kept


def numbers(part: Any) -> Any:
    """A part as numpy, a row for each channel, wherever the model left it."""
    import numpy as np

    if hasattr(part, "detach"):
        part = part.detach().cpu().numpy()
    return np.asarray(part, dtype=np.float32)


class DemucsSeparation:
    """A Demucs model run in this process."""

    name = "demucs"

    def __init__(self, model: str | None = None, *, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.from_env()
        # Nothing built in fills the slot, so with none chosen the settings are whoever's is asked for.
        chosen = self._settings.adapter(SLOT) in (None, self.name)
        self.model = model or (self._settings.model(SLOT) if chosen else None) or DEFAULT_MODEL
        self.options = options(self._settings, chosen)
        self._loaded: Any = None

    # The model.

    def device(self, torch: Any) -> str:
        """What the model runs on: what was asked for, or the best there is."""
        have = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
        wanted = self._settings.device
        if wanted == "auto":
            return next(name for name in DEVICES if have[name])
        if wanted not in have:
            raise SeparationError(f"the demucs adapter runs on one of {', '.join(DEVICES)}, not on {wanted!r}")
        if not have[wanted]:
            raise SeparationError(f"this PyTorch cannot use {wanted!r} on this machine")
        return wanted

    def loaded(self) -> Any:
        if self._loaded is None:
            # What PyTorch has not written for Apple's graphics processors yet runs on the CPU.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            try:
                import torch
                from demucs import api
            except ImportError:
                raise SeparationError("the demucs separation adapter needs Demucs and PyTorch: "
                                      'pip install "demucs>=4.1" torch') from None
            device = self.device(torch)
            try:
                separator = api.Separator(model=self.model, device=device, shifts=int(self.options["shifts"]),
                                          overlap=float(self.options["overlap"]), progress=False)
            except Exception as error:  # noqa: BLE001 - whatever a model fails to load with, say it plainly
                raise SeparationError(f"the separation model {self.model} could not be loaded: {error}") from None
            keeps = [str(one) for one in separator.model.sources]
            if THEIRS not in keeps or len(keeps) < 2:
                raise SeparationError(f"the separation model {self.model} keeps no voice apart: its parts are "
                                      f"{', '.join(keeps) or 'none'}")
            if separator.audio_channels != CHANNELS:
                raise SeparationError(f"the separation model {self.model} hears {separator.audio_channels} "
                                      f"channels, and the demucs adapter runs the ones that hear {CHANNELS}")
            self._loaded = (torch, separator)
        return self._loaded

    # Pulling apart.

    def separate(self, source: Path, folder: Path) -> dict[str, Path]:
        import numpy as np

        from constructive_media.audio import pcm

        source, folder = Path(source), Path(folder)
        wide = pcm.channels(source) > 1
        with gpu.alone(self._settings):
            torch, separator = self.loaded()
            rate = int(separator.samplerate)
            sound = pcm.read(source, rate=rate)                 # two channels, a row a moment
            if not len(sound):
                raise SeparationError(f"{source.name} holds no sound")
            try:
                _, made = separator.separate_tensor(torch.from_numpy(np.ascontiguousarray(sound.T)), rate)
            except Exception as error:  # noqa: BLE001
                raise SeparationError(f"the separation model {self.model} could not pull {source.name} apart: "
                                      f"{error}") from None
            others = {str(name): numbers(part) for name, part in made.items()}
        voice = others.pop(THEIRS)
        if self.options["parts"] == "all":
            parts = {VOICE: voice, **others}
        else:
            parts = {VOICE: voice, REST: sum(others.values())}
        # A recording of one channel comes back as one: both of the model's are the same sound.
        return {name: pcm.write(folder / f"{name}.wav", part.T if wide else part.mean(axis=0), rate)
                for name, part in parts.items()}


def adapters() -> list[Adapter]:
    """What this package adds to the engine."""
    return [Adapter(
        slot=SLOT, name=DemucsSeparation.name, factory=lazy("media_engine_demucs:DemucsSeparation"), license=LICENSE,
        summary="A voice taken out from over music, with Demucs on this machine; the weights are for research",
    )]
