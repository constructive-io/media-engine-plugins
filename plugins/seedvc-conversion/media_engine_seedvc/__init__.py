"""Voice conversion for Constructive Media, through Seed-VC.

Fills the engine's `voice_conversion` slot under the name `seedvc`: what one
voice said is said again in the voice of a clip, with the words, the timing
and the tune of the first and the sound of the second. It needs no training
and no words written down: a clip of a few seconds is enough.

    media revoice take.wav take-as-ada.wav --voice ada
    media talk --voice ada --quick af_heart

Seed-VC is licensed GPL-3.0, code and weights, so this adapter is too, and
that is why it lives here and not in the engine.

What it is run with is read from `CONSTRUCTIVE_MEDIA_EXTRA_VOICE_CONVERSION`,
a JSON object. The defaults are

    {"steps": 10, "guidance": 0.7, "length": 1.0, "fp16": true}

`steps` is how many passes the model makes over a sentence. More takes
longer, and the package's own choice is 30; from 4 to 30 a speaker model heard
no difference in whose voice it was, so listen before paying for more.
`guidance` is how hard it is pulled towards the clip, `length` makes what is
said longer or shorter than it was, and `fp16` runs it in half precision.

    Copyright (C) 2026 Dan Lynch <pyramation@gmail.com>
    Copyright (C) 2026 Constructive <developers@constructive.io>

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License, version 3, as published
    by the Free Software Foundation. It is distributed WITHOUT ANY WARRANTY;
    see the LICENSE file beside this package.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import threading
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from constructive_media.adapters import Adapter, lazy
from constructive_media.conversion import ConversionError
from constructive_media.settings import Settings

__version__ = "0.1.0"

LICENSE = "GPL-3.0-only"
SLOT = "voice_conversion"

#: Where the weights come from. The package fetches them the first time they are asked for.
MODEL = "Plachta/Seed-VC"
DEFAULTS: dict[str, Any] = {"steps": 10, "guidance": 0.7, "length": 1.0, "fp16": True}
SETTING = "CONSTRUCTIVE_MEDIA_EXTRA_VOICE_CONVERSION"

#: The rate the model works at. Sound is handed to it at this rate and comes back at it.
RATE = 22_050
#: The model hears this much of a clip and no more.
LONGEST_CLIP = 25.0
#: How many clips are kept ready. A conversation is said in one voice, a scene in a few.
CLIPS_KEPT = 8

#: The package holds one set of models, and more than one thread may ask for a sentence.
_ONE_AT_A_TIME = threading.Lock()
_MARK = "_kept_by_media_engine"


def adapters() -> list[Adapter]:
    """What this plugin gives the engine."""
    return [
        Adapter(
            slot=SLOT,
            name="seedvc",
            factory=lazy("media_engine_seedvc:SeedVC"),
            license=LICENSE,
            summary="What was said, said again in the voice of a clip, by Seed-VC on this machine",
        )
    ]


def options(settings: Settings) -> dict[str, Any]:
    """What the model is run with: the defaults, and over them what the setting says."""
    written = settings.extra(SLOT)
    if not written:
        return dict(DEFAULTS)
    try:
        given = json.loads(written)
    except ValueError as error:
        raise ConversionError(f"{SETTING} is not JSON: {error}") from None
    if not isinstance(given, dict):
        raise ConversionError(f'{SETTING} should be a JSON object, such as {{"steps": 4}}')
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise ConversionError(f"{SETTING} says {', '.join(unknown)}, which the seedvc adapter does not take. "
                              f"It takes {', '.join(DEFAULTS)}")
    chosen = {**DEFAULTS, **given}
    steps, fp16 = chosen["steps"], chosen["fp16"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 200:
        raise ConversionError(f"the steps in {SETTING} are a whole number from 1 to 200, not {steps!r}")
    if not isinstance(fp16, bool):
        raise ConversionError(f"fp16 in {SETTING} is true or false, not {fp16!r}")
    for name, least, most in (("guidance", 0.0, 5.0), ("length", 0.25, 4.0)):
        value = chosen[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not least <= value <= most:
            raise ConversionError(f"the {name} in {SETTING} is a number from {least} to {most}, not {value!r}")
        chosen[name] = float(value)
    return chosen


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Keep what the package prints off the screen, and its counting of steps: a command's output is its own."""
    with open(os.devnull, "w") as nowhere, contextlib.redirect_stdout(nowhere), contextlib.redirect_stderr(nowhere):
        yield


def package() -> tuple[Any, Any]:
    """Seed-VC's own code, made to behave as a part of something else: its `api` module and its sound class.

    Left to itself the package does three things a library should not. On
    being imported it tells Hugging Face to keep models in a `checkpoints`
    folder of wherever the program was started, and turns every warning in
    the process off. It fetches its own weights into that folder too. And it
    loads all of its models again for every recording it is handed, which
    takes longer than saying a sentence does.

    So: Hugging Face is imported first and its setting put back, which
    leaves the models where yours are kept. The warnings are put back. The
    weights are fetched to the same place as the rest. And the models are
    loaded once and kept.
    """
    try:
        hub = importlib.import_module("huggingface_hub")
        # It reads where models are kept when this part of it is first imported, which is not until it is asked for.
        importlib.import_module("huggingface_hub.constants")
    except ImportError as error:
        raise ConversionError(f"the seedvc adapter needs seed-vc: pip install seed-vc ({error})") from None
    was = os.environ.get("HF_HUB_CACHE")
    try:
        with warnings.catch_warnings(), quiet():
            warnings.simplefilter("ignore")
            api = importlib.import_module("seed_vc.api")
            loader = importlib.import_module("seed_vc.inference_realtime")
            sound = importlib.import_module("seed_vc.Models.audio").AudioData
    except ImportError as error:
        raise ConversionError(f"the seedvc adapter needs seed-vc: pip install seed-vc ({error})") from None
    finally:
        if was is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = was
    if getattr(api, _MARK, False):
        return api, sound
    for module, name in ((api, "inference"), (api, "load_models_realtime"), (loader, "load_custom_model_from_hf")):
        if not callable(getattr(module, name, None)):
            raise ConversionError(f"this seed-vc has no {module.__name__}.{name}: the seedvc adapter was written "
                                  "for seed-vc 0.4")

    def fetched(repo_id: str, model_filename: str = "pytorch_model.bin", config_filename: str | None = None) -> Any:
        weights = hub.hf_hub_download(repo_id=repo_id, filename=model_filename)
        if config_filename is None:
            return weights
        return weights, hub.hf_hub_download(repo_id=repo_id, filename=config_filename)

    loads, kept = api.load_models_realtime, {}

    def once(args: Any) -> Any:
        asked = (getattr(args, "f0_condition", None), getattr(args, "checkpoint", None), getattr(args, "fp16", None))
        if asked not in kept:
            kept[asked] = loads(args)
        return kept[asked]

    loader.load_custom_model_from_hf = fetched
    api.load_models_realtime = once
    setattr(api, _MARK, True)
    return api, sound


class SeedVC:
    """Seed-VC, run in this process."""

    name = "seedvc"
    model = MODEL

    def __init__(self, model: str | None = None, *, settings: Settings | None = None) -> None:
        settings = settings or Settings.from_env()
        asked = model or settings.model(SLOT)
        if asked and asked != MODEL:
            raise ConversionError(f"the seedvc adapter runs one model, {MODEL}, and was asked for {asked}")
        self.options = options(settings)
        self._package: tuple[Any, Any] | None = None
        self._clips: dict[tuple[str, int, int], Any] = {}

    def loaded(self) -> tuple[Any, Any]:
        if self._package is None:
            self._package = package()
        return self._package

    def convert(self, source: Path, target: Path, *, like: Path) -> Path:
        """Say what `source` says in the voice `like` is in, as a WAV file at `target`."""
        from constructive_media.audio import pcm

        source, target, like = Path(source), Path(target), Path(like)
        said = pcm.read(source, rate=RATE, stereo=False)
        if not len(said):
            raise ConversionError(f"{source.name} has no sound in it")
        with _ONE_AT_A_TIME:
            api, sound = self.loaded()
            clip = self._clip(like, sound)
            try:
                with quiet(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    made = api.inference(
                        source=_handed(said, sound), target=clip, diffusion_steps=self.options["steps"],
                        length_adjust=self.options["length"], inference_cfg_rate=self.options["guidance"],
                        fp16=self.options["fp16"],
                    )
            except Exception as error:  # noqa: BLE001 - whatever a model raises, it is the same news
                raise ConversionError(f"Seed-VC could not say {source.name} again: {error}") from error
        rate, samples = _given_back(made)
        if not len(samples):
            raise ConversionError(f"Seed-VC said nothing for {source.name}")
        return pcm.write(target, samples, rate, bits=16)

    def _clip(self, like: Path, sound: Any) -> Any:
        """The clip as the model takes it, kept for the next sentence."""
        from constructive_media.audio import pcm

        about = like.stat()
        which = (str(like.resolve()), about.st_mtime_ns, about.st_size)
        if which not in self._clips:
            heard = pcm.read(like, rate=RATE, stereo=False)[: int(LONGEST_CLIP * RATE)]
            if not len(heard):
                raise ConversionError(f"{like.name} has no sound in it")
            while len(self._clips) >= CLIPS_KEPT:
                self._clips.pop(next(iter(self._clips)))
            self._clips[which] = _handed(heard, sound)
        return self._clips[which]


def _handed(samples: Any, sound: Any) -> Any:
    """Sound as the package takes it: whole numbers of 16 bits, with what it is written beside them."""
    import numpy as np

    whole = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    return sound(whole, None, len(whole) / RATE, len(whole), RATE, None)


def _given_back(made: Any) -> tuple[int, Any]:
    """The rate and the samples of what the package gave back, which has been two different things."""
    import numpy as np

    rate, samples = made if isinstance(made, tuple) else (made.sample_rate, made.samples)
    samples = np.asarray(samples)
    if samples.dtype.kind in "iu":
        return int(rate), samples.astype(np.float32) / 32768
    return int(rate), np.clip(samples.astype(np.float32), -1.0, 1.0)
