"""Video from words or from a still for Constructive Media, through LTX-Video.

Fills the engine's `video` slot under the name `ltx`. The engine's own video
adapter starts from words and nothing else. This one starts from words, or
from a picture and words: the picture is the first frame, and the words say
what happens next.

    media generate video "fog drifting over the water" --from harbour.png --adapter ltx
    media generate video "fog drifting over a harbour at dawn" --seconds 4 --adapter ltx

The code here is MIT, and what it imports is permissively licensed too. The
model is not: Lightricks publish each version of LTX-Video under terms of
their own, which forbid some uses outright and, for the later versions, ask
companies above a size to buy a license. That is why this adapter lives
here and not in the engine, and why it says so wherever adapters are
listed. The README beside this package has the terms, version by version.

The model is `CONSTRUCTIVE_MEDIA_MODEL_VIDEO` and what a piece is made with
is `CONSTRUCTIVE_MEDIA_EXTRA_VIDEO`, a JSON object. Both are read only when
this is the adapter chosen for video, so that the engine's own adapter can
keep its settings on the same machine. The defaults are

    {"steps": 25, "fps": 24, "guidance": 3.0, "rough": 12,
     "negative": "worst quality, inconsistent motion, blurry, jittery, distorted"}

`steps` is how many passes the model makes, and `rough` is how hard a
starting picture is squeezed through a video encoder before the model sees
it, from 1 to 31, or 0 to leave it as it is. The model learned from video,
and a picture cleaner than any frame of video it holds still.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from constructive_media import ffmpeg, gpu
from constructive_media.adapters import Adapter, lazy
from constructive_media.generate import GenerateError, Request
from constructive_media.generate.diffusers import written
from constructive_media.settings import Settings

__version__ = "0.1.0"

#: The model's terms, which are not one license but one for each version: see the README.
LICENSE = "LicenseRef-Lightricks-LTX-Video"
SLOT = "video"
SETTING = "CONSTRUCTIVE_MEDIA_EXTRA_VIDEO"

#: The first version whose terms are not for research alone, and the one this was tried with.
DEFAULT_MODEL = "Lightricks/LTX-Video-0.9.5"
#: A size the model does well at. Sides are multiples of 32.
WIDTH, HEIGHT = 704, 480
SIDE = 32
SECONDS = 2.0
#: The model writes eight frames at a time after the first, so a piece is 8n + 1 frames long.
FRAMES_TOGETHER = 8
LEAST_FRAMES = 9
DEFAULTS: dict[str, Any] = {
    "steps": 25, "fps": 24, "guidance": 3.0, "rough": 12,
    "negative": "worst quality, inconsistent motion, blurry, jittery, distorted",
}
#: How rough a picture can be made: what ffmpeg's own MPEG-4 encoder takes for its quality.
ROUGHEST = 31
#: What the part that turns numbers into pictures is asked to clean up as it does.
DECODE_AT, DECODE_NOISE = 0.05, 0.025
DEVICES = ("mps", "cuda", "cpu")


def frames_for(seconds: float, fps: int) -> int:
    """How many frames the model is asked for: the nearest count it can write, and never too few to move."""
    return max(LEAST_FRAMES, round(seconds * fps / FRAMES_TOGETHER) * FRAMES_TOGETHER + 1)


def size_like(width: int, height: int) -> tuple[int, int]:
    """A size the model takes, of about the area it does well at, in the shape of a picture."""
    scale = math.sqrt(WIDTH * HEIGHT / (width * height))
    return max(SIDE, round(width * scale / SIDE) * SIDE), max(SIDE, round(height * scale / SIDE) * SIDE)


def first_frame(picture: Path, width: int, height: int, rough: int) -> bytes:
    """A picture as the pixels the model starts from: filled to the size, cut to it, and roughened.

    The picture is never squashed. It is scaled until it covers the size and
    what hangs over the edges is cut off, evenly from both sides.
    """
    fitted = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height}"
    try:
        if not rough:
            return ffmpeg.run(ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", picture, "-vf", fitted,
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-", binary=True).stdout
        with tempfile.TemporaryDirectory(prefix="first-frame-") as folder:
            squeezed = Path(folder) / "frame.mp4"
            # The encoder every ffmpeg has, whatever it was built with.
            ffmpeg.run(ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", picture, "-vf", fitted,
                       "-frames:v", "1", "-c:v", ffmpeg.FALLBACK_ENCODER, "-q:v", str(rough),
                       "-pix_fmt", "yuv420p", squeezed)
            return ffmpeg.frame(squeezed, 0.0)
    except ffmpeg.FfmpegError as error:
        raise GenerateError(f"the picture {picture} could not be read: {error}") from None


class LtxVideo:
    """An LTX-Video model run in this process through diffusers."""

    name = "ltx"

    def __init__(self, model: str | None = None, *, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.from_env()
        chosen = self._settings.adapter(SLOT) == self.name
        self.model = model or (self._settings.model(SLOT) if chosen else None) or DEFAULT_MODEL
        self.options = dict(DEFAULTS)
        extra = self._settings.extra(SLOT) if chosen else None
        if extra:
            try:
                given = json.loads(extra)
            except ValueError as error:
                raise GenerateError(f"{SETTING} is not JSON: {error}") from None
            if not isinstance(given, dict):
                raise GenerateError(f'{SETTING} should be a JSON object, such as {{"steps": 25}}')
            unknown = sorted(set(given) - set(DEFAULTS))
            if unknown:
                raise GenerateError(f"{SETTING} has {', '.join(unknown)}, which the ltx adapter does not know; "
                                    f"it knows: {', '.join(DEFAULTS)}")
            self.options.update(given)
        rough = self.options["rough"]
        if isinstance(rough, bool) or not isinstance(rough, int) or not 0 <= rough <= ROUGHEST:
            raise GenerateError(f"the rough in {SETTING} is a whole number from 0, which leaves a picture as it "
                                f"is, to {ROUGHEST}: not {rough!r}")
        self._loaded: Any = None

    # The model.

    def device(self, torch: Any) -> str:
        """What the model runs on: what was asked for, or the best there is."""
        have = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
        wanted = self._settings.device
        if wanted == "auto":
            return next(name for name in DEVICES if have[name])
        if wanted not in have:
            raise GenerateError(f"the ltx adapter runs on one of {', '.join(DEVICES)}, not on {wanted!r}")
        if not have[wanted]:
            raise GenerateError(f"this PyTorch cannot use {wanted!r} on this machine")
        return wanted

    def loaded(self) -> Any:
        if self._loaded is None:
            # What PyTorch has not written for Apple's graphics processors yet runs on the CPU.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            try:
                import torch
                from diffusers import LTXConditionPipeline
            except ImportError:
                raise GenerateError("the ltx video adapter needs PyTorch and a diffusers that has LTX-Video: "
                                    'pip install torch "diffusers>=0.33" transformers sentencepiece') from None
            device = self.device(torch)
            # A command's output is its own: the libraries draw their progress over it otherwise.
            for library in ("diffusers", "transformers"):
                with contextlib.suppress(Exception):
                    importlib.import_module(f"{library}.utils.logging").disable_progress_bar()
            try:
                pipeline = LTXConditionPipeline.from_pretrained(
                    self.model, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
                ).to(device)
                pipeline.set_progress_bar_config(disable=True)
            except Exception as error:  # noqa: BLE001 - whatever a model fails to load with, say it plainly
                raise GenerateError(f"the video model {self.model} could not be loaded: {error}") from None
            if getattr(getattr(pipeline.scheduler, "config", None), "use_dynamic_shifting", False):
                raise GenerateError(f"the video model {self.model} is laid out as LTX-Video was before 0.9.5, which "
                                    f"the ltx adapter does not run: use 0.9.5 or a later one, such as {DEFAULT_MODEL}")
            self._loaded = (torch, pipeline)
        return self._loaded

    # Making.

    def size(self, request: Request) -> tuple[int, int]:
        """The size of the piece: what was asked for, or the picture's own shape, or the model's own size."""
        if request.width or request.height or request.image is None:
            return request.width or WIDTH, request.height or HEIGHT
        picture = Path(request.image)
        try:
            described = ffmpeg.probe(picture) if picture.is_file() else None
        except ffmpeg.FfmpegError as error:
            raise GenerateError(f"the picture {picture} could not be read: {error}") from None
        if described is None or not described.width or not described.height:
            raise GenerateError(f"the picture to start from is not one that can be read: {picture}")
        return size_like(described.width, described.height)

    def check(self, request: Request) -> None:
        """Refuse what cannot be made, before a model is loaded."""
        if request.medium != "video":
            raise GenerateError("the ltx adapter makes video, not pictures")
        width, height = self.size(request)
        if width % SIDE or height % SIDE:
            raise GenerateError(f"{width}x{height} cannot be made: each side is a multiple of {SIDE}, "
                                f"such as {max(SIDE, round(width / SIDE) * SIDE)}x"
                                f"{max(SIDE, round(height / SIDE) * SIDE)}")

    def make(self, request: Request, target: Path) -> Path:
        import numpy as np

        self.check(request)
        width, height = self.size(request)
        fps = int(self.options["fps"])
        asked: dict[str, Any] = {}
        if request.image is not None:
            from PIL import Image

            pixels = first_frame(Path(request.image), width, height, int(self.options["rough"]))
            if len(pixels) != width * height * 3:
                raise GenerateError(f"the picture to start from is not one that can be read: {request.image}")
            asked["image"] = Image.frombytes("RGB", (width, height), pixels)
        with gpu.alone(self._settings):
            torch, pipeline = self.loaded()
            if request.seed is not None:
                asked["generator"] = torch.Generator(device="cpu").manual_seed(int(request.seed))
            try:
                made = pipeline(
                    prompt=request.prompt, negative_prompt=self.options["negative"] or None,
                    height=height, width=width, num_frames=frames_for(request.seconds or SECONDS, fps),
                    frame_rate=fps, num_inference_steps=int(request.steps or self.options["steps"]),
                    guidance_scale=float(self.options["guidance"]),
                    decode_timestep=DECODE_AT, decode_noise_scale=DECODE_NOISE, output_type="np", **asked,
                ).frames[0]
            except Exception as error:  # noqa: BLE001
                raise GenerateError(f"the video model {self.model} could not make the piece: {error}") from None
        frames = np.asarray(made)
        if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
            raise GenerateError(f"the video model {self.model} answered with nothing to show")
        return written(frames, fps, Path(target))


def adapters() -> list[Adapter]:
    """What this package adds to the engine."""
    return [Adapter(
        slot=SLOT, name=LtxVideo.name, factory=lazy("media_engine_ltx:LtxVideo"), license=LICENSE,
        summary="Video from words or from a still, with LTX-Video on this machine; the model has terms of its own",
    )]
