"""LatentSync, run from a checkout of it on whatever this machine has.

This file is not imported by the adapter. The adapter hands it to the
Python that has what LatentSync needs, in a folder made for the one run:

    python runner.py --repo ~/LatentSync --model ByteDance/LatentSync-1.5 \
        --config configs/unet/stage2.yaml --faces ~/models/faces \
        --video video.mp4 --speech speech.wav --out synced.mp4 --report report.json

The folder it runs in has the checkout's `configs` and `latentsync` linked
into it. LatentSync reads both by names that start where it is run, and
writes what it is working on to a `temp` folder there, so run this way
nothing is written into the checkout and two runs do not write over each
other. The video, the speech and the result are named without their
folder, because LatentSync hands those names to a shell with nothing
around them.

It makes the calls LatentSync's own `scripts/inference.py` makes, in the
same order, DeepCache among them when it is asked for (`--deepcache`, as
that script has it: every third step is worked in full and the two between
reuse most of it). One thing is its own everywhere: the face finder's models
are kept in the folder it is told, not in the checkout. That script is
written for an NVIDIA GPU and names one throughout. Where there is one,
nothing else here stands in for anything. Where there is not, each of
these is stood in for as LatentSync is loaded, and its files are left as
they were:

- `decord`, which it imports and which has no build for Apple silicon. It
  uses it for one thing when it moves a mouth, reading the speech, and the
  speech is read here instead.
- the device, where the script says `"cuda"`: the one this machine has.
- half precision, which it asks an NVIDIA GPU about: off one it is full
  precision. On Apple's graphics processors half precision runs, and
  paints coloured specks over the mouth.
- the face finder, which is told to run on an NVIDIA GPU and nothing else:
  it runs on the CPU, where it takes a few seconds for a clip.
- the pipeline, which builds its face tools on `"cuda"` whatever it was
  moved to.

Nothing here is imported from the engine: the Python that runs this is
LatentSync's, and need not have the engine in it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
import types
from typing import Any, Callable

DEVICES = ("cuda", "mps", "cpu")

#: The U-Net as its makers publish it, in the folder of a model on Hugging Face.
UNET = "latentsync_unet.pt"
#: The speech model each width of U-Net listens through.
LISTENS = {384: "tiny.pt", 768: "small.pt"}
#: The picture model LatentSync's own script fetches.
VAE = "stabilityai/sd-vae-ft-mse"

NO_FACE = "Face not detected"


class Refusal(Exception):
    """Something that cannot be done, said in words a person can act on."""

    def __init__(self, words: str, kind: str = "refused") -> None:
        super().__init__(words)
        self.kind = kind


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="the LatentSync checkout")
    parser.add_argument("--model", required=True, help="where the weights are fetched from on Hugging Face")
    parser.add_argument("--config", required=True, help="the U-Net's config, from the folder this runs in")
    parser.add_argument("--faces", required=True, help="the folder the face finder keeps its models in")
    parser.add_argument("--unet", help="the U-Net's weights as a file, in place of the model's own")
    parser.add_argument("--video", required=True)
    parser.add_argument("--speech", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True, help="where what happened is written, as JSON")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--deepcache", action="store_true", help="reuse most of each step for the two after it")
    return parser.parse_args(argv)


# What this machine has.


def pick(asked: str, have: dict[str, bool]) -> str:
    """The device to run on: the one asked for, or the best there is."""
    if asked == "auto":
        return next(name for name in DEVICES if have.get(name))
    if asked not in DEVICES:
        raise Refusal(f"LatentSync runs on one of {', '.join(DEVICES)}, not on {asked!r}")
    if not have.get(asked):
        raise Refusal(f"this PyTorch cannot use {asked!r} on this machine")
    return asked


def having(torch: Any) -> dict[str, bool]:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return {"cuda": bool(torch.cuda.is_available()), "mps": bool(mps is not None and mps.is_available()), "cpu": True}


def halved(torch: Any, device: str) -> bool:
    """Whether the weights run at half precision: what LatentSync's own script asks, of an NVIDIA GPU only."""
    return device == "cuda" and torch.cuda.get_device_capability()[0] > 7


# What is stood in for.


def without_decord() -> bool:
    """Put something where `decord` would be when it is not installed, and say whether that was done."""
    try:
        importlib.import_module("decord")
    except Exception:  # noqa: BLE001 - not installed, or installed and unable to load its library
        stand_in = types.ModuleType("decord")

        class Absent:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("decord is not installed")

        stand_in.AudioReader = Absent    # type: ignore[attr-defined]
        stand_in.VideoReader = Absent    # type: ignore[attr-defined]
        sys.modules["decord"] = stand_in
        return True
    return False


def read_audio(audio_path: str, audio_sample_rate: int = 16000) -> Any:
    """The speech as LatentSync's own reader gives it: one channel, as numbers from -1 to 1."""
    import wave

    import numpy
    import torch

    with wave.open(audio_path, "rb") as held:
        rate, channels, width = held.getframerate(), held.getnchannels(), held.getsampwidth()
        raw = held.readframes(held.getnframes())
    if width != 2:
        raise Refusal(f"{audio_path} is not the 16-bit recording this was handed")
    samples = numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != audio_sample_rate:
        import librosa

        samples = librosa.resample(samples, orig_sr=rate, target_sr=audio_sample_rate)
    return torch.from_numpy(numpy.ascontiguousarray(samples))


def faces_kept_in(detector: Any, folder: str, device: str) -> None:
    """Have the face finder keep its models in `folder`, and run where this machine can run it."""
    real = detector.FaceAnalysis

    def analysis(*args: Any, **asked: Any) -> Any:
        asked["root"] = folder
        if device != "cuda":
            asked["providers"] = ["CPUExecutionProvider"]
        return real(*args, **asked)

    detector.FaceAnalysis = analysis
    if device != "cuda":
        # LatentSync turns the name of an NVIDIA GPU into its number. Below nought is the CPU.
        detector.cuda_to_int = lambda _name: -1


def tools_on(pipeline: Any, detector: Any, restorer: Callable[..., Any], on: str, precision: Any) -> None:
    """Have the pipeline build its face tools on the device `on`: it builds them on "cuda" wherever it was moved."""
    real = pipeline.ImageProcessor

    def processor(resolution: int = 512, device: str = "cuda", mask_image: Any = None) -> Any:
        made = real(resolution, device=on, mask_image=mask_image)        # whatever `device` the pipeline named
        # Where a face goes in the frame is worked out at half precision unless told otherwise.
        made.restorer = restorer(resolution=resolution, device=on, dtype=precision)
        if getattr(made, "face_detector", None) is None:
            # On the CPU LatentSync builds no face finder at all. With the stand-ins above there is one.
            made.face_detector = detector.FaceDetector(device=on)
        return made

    pipeline.ImageProcessor = processor


# The weights.


def fetched(model: str, name: str) -> str:
    """A file of a model on Hugging Face, from where such files are kept on this machine, fetched when not there."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise Refusal("LatentSync's weights are fetched with huggingface_hub, which this Python does not have: "
                      "pip install huggingface-hub", "missing") from None
    try:
        return str(hf_hub_download(model, name))
    except Exception as error:  # noqa: BLE001 - no network, no such model, no such file
        raise Refusal(f"{name} of {model} could not be fetched: {error}", "weights") from None


def listens_through(repo: str, model: str, width: int) -> str:
    """The speech model a U-Net of this width listens through: the checkout's own, or the published one."""
    if width not in LISTENS:
        raise Refusal(f"a U-Net that listens {width} wide is not one LatentSync has a speech model for: "
                      f"it has them for {', '.join(str(one) for one in LISTENS)}")
    own = os.path.join(repo, "checkpoints", "whisper", LISTENS[width])
    return own if os.path.isfile(own) else fetched(model, "whisper/" + LISTENS[width])


# The run.


def run(asked: argparse.Namespace) -> dict[str, Any]:
    # What PyTorch has not written for Apple's graphics processors yet runs on the CPU.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.path.insert(0, asked.repo)
    began = time.perf_counter()
    try:
        import torch
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKL, DDIMScheduler
        from omegaconf import OmegaConf

        device = pick(asked.device, having(torch))
        stood_in = without_decord()
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines import lipsync_pipeline
        from latentsync.utils import face_detector
        from latentsync.utils.affine_transform import AlignRestore
        from latentsync.whisper.audio2feature import Audio2Feature
    except ImportError as error:
        raise Refusal(f"this Python does not have what LatentSync needs: {error}", "missing") from None
    if asked.deepcache:
        try:
            from DeepCache import DeepCacheSDHelper
        except ImportError:
            raise Refusal("deepcache was asked for, and this Python does not have it: pip install DeepCache",
                          "missing") from None

    precision = torch.float16 if halved(torch, device) else torch.float32
    if stood_in:
        lipsync_pipeline.read_audio = read_audio
    os.makedirs(asked.faces, exist_ok=True)
    faces_kept_in(face_detector, asked.faces, device)
    if device != "cuda":
        tools_on(lipsync_pipeline, face_detector, AlignRestore, device, torch.float32)

    config = OmegaConf.load(asked.config)
    unet_path = asked.unet or fetched(asked.model, UNET)
    whisper_path = listens_through(asked.repo, asked.model, int(config.model.cross_attention_dim))

    # From here down, LatentSync's scripts/inference.py with the device this machine has.
    scheduler = DDIMScheduler.from_pretrained("configs")
    audio_encoder = Audio2Feature(
        model_path=whisper_path, device=device, num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=precision)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(config.model), unet_path, device="cpu")
    unet = unet.to(dtype=precision)
    pipeline = lipsync_pipeline.LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler,
    ).to(device)
    if asked.deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    set_seed(asked.seed)
    loaded = time.perf_counter()

    if os.path.exists(asked.out):
        os.remove(asked.out)
    pipeline(
        video_path=asked.video, audio_path=asked.speech, video_out_path=asked.out,
        num_frames=config.data.num_frames, num_inference_steps=asked.steps, guidance_scale=asked.guidance,
        weight_dtype=precision, width=config.data.resolution, height=config.data.resolution,
        mask_image_path=config.data.mask_image_path, temp_dir="temp",
    )
    # LatentSync hands the last step to ffmpeg and does not look at what came of it.
    if not os.path.isfile(asked.out) or not os.path.getsize(asked.out):
        raise Refusal("LatentSync ran to its end and wrote no video: what ffmpeg said is above")
    return {
        "ok": True, "device": device, "precision": str(precision).rsplit(".", 1)[-1],
        "resolution": int(config.data.resolution), "steps": asked.steps, "deepcache": bool(asked.deepcache),
        "stood_in": device != "cuda",
        "loading": round(loaded - began, 2), "moving": round(time.perf_counter() - loaded, 2),
    }


def said(error: BaseException) -> dict[str, Any]:
    """What went wrong, for whoever asked."""
    if isinstance(error, Refusal):
        return {"ok": False, "kind": error.kind, "error": str(error)}
    if NO_FACE in str(error):
        return {"ok": False, "kind": "no_face", "error": str(error)}
    if isinstance(error, MemoryError) or "out of memory" in str(error).lower():
        return {"ok": False, "kind": "memory", "error": str(error)}
    return {"ok": False, "kind": "failed", "error": f"{type(error).__name__}: {error}"}


def main(argv: list[str] | None = None) -> int:
    asked = arguments(argv)
    try:
        told = run(asked)
    except Exception as error:  # noqa: BLE001 - whatever it was is written down for whoever asked
        traceback.print_exc()
        told = said(error)
    with open(asked.report, "w", encoding="utf-8") as report:
        json.dump(told, report)
    return 0 if told["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
