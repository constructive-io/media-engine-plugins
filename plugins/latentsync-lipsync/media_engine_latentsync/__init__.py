"""A face in a video made to say a recording, for Constructive Media, through LatentSync.

Fills the engine's `lip_sync` slot under the name `latentsync`. A piece of
video with a face in it and a recording of speech go in, and the video
comes back with the mouth saying the recording.

    media lipsync face.mp4 line.wav out/face-says-line.mp4 --adapter latentsync

LatentSync is not a package that can be installed. It is a repository that
is run from where it was cloned, with a long list of what it needs, and its
script is written for an NVIDIA GPU. So this adapter installs none of it.
It is told where a checkout of LatentSync is and which Python has what
LatentSync needs, and it runs LatentSync there, a run at a time, in a
folder of its own. The checkout's files are left as they are: what has to
be different off an NVIDIA GPU is stood in for as the model is loaded
(`runner.py`, beside this file, says what).

    CONSTRUCTIVE_MEDIA_EXTRA_LIP_SYNC='{"repo": "~/LatentSync", "python": "~/LatentSync/.venv/bin/python"}'

The code here is MIT. What it runs is not all so. LatentSync's code is
Apache-2.0; its weights are published under the Open RAIL++-M License,
which says what they may not be used for; and the models its face finder
fetches, from InsightFace, are in their makers' words "for non-commercial
research purposes only". That last is why this adapter lives here and not
in the engine, and why it says so wherever adapters are listed. The README
beside this package has each of them in their makers' words.

The model is `CONSTRUCTIVE_MEDIA_MODEL_LIP_SYNC`, the name of a model on
Hugging Face, and what a face is moved with is
`CONSTRUCTIVE_MEDIA_EXTRA_LIP_SYNC`, a JSON object. Both are read when this
is the adapter chosen for lip sync, or when none is and this one is asked
for by name. Besides `repo` and `python`, the defaults are

    {"config": "configs/unet/stage2.yaml", "steps": 20, "guidance": 1.5, "seed": 1247, "deepcache": false}

`config` is the file in the checkout that says how big a face the model
works on. `steps` is how many times over each stretch of the picture is
worked, `guidance` is how hard the mouth is held to the speech, and `seed`
is where the model's chance starts: the same seed moves the same mouth the
same way. `checkpoint` names a file of weights to run in place of the
model's own. `deepcache` works every third step in full and has the two
between reuse most of it, as LatentSync's own `inference.sh` does; it needs
the DeepCache package in LatentSync's Python.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from constructive_media import ffmpeg, gpu
from constructive_media.adapters import Adapter, lazy
from constructive_media.lipsync import LipSyncError
from constructive_media.settings import Settings

__version__ = "0.1.0"

#: Not the name of one license: what is run comes under several, and one of them is for research. See the README.
LICENSE = "LicenseRef-LatentSync-OpenRAIL-weights-and-research-only-face-models"
SLOT = "lip_sync"
SETTING = "CONSTRUCTIVE_MEDIA_EXTRA_LIP_SYNC"

#: The model whose mouths hold still between frames, at the size that fits an ordinary machine.
DEFAULT_MODEL = "ByteDance/LatentSync-1.5"
DEFAULTS: dict[str, Any] = {
    "repo": None, "python": None, "checkpoint": None,
    "config": "configs/unet/stage2.yaml", "steps": 20, "guidance": 1.5, "seed": 1247, "deepcache": False,
}
MOST_STEPS = 100
MOST_GUIDANCE = 10.0

#: What is handed to LatentSync's own Python.
RUNNER = Path(__file__).with_name("runner.py")
#: What a checkout of LatentSync has in it, and what is linked into the folder a run is made in.
HAS = ("latentsync/pipelines/lipsync_pipeline.py", "configs/scheduler_config.json")
LINKED = ("configs", "latentsync")
#: Where a checkout's own Python is looked for, when none is named.
PYTHONS = (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe", "venv/Scripts/python.exe")
#: The speech as LatentSync hears it.
RATE = 16_000
#: How much of the video is handed over past the end of the speech, in seconds.
PAST = 1.0
#: How much of what LatentSync printed is shown when it fails.
SHOWN = 12
#: Of the environment this is run in, what would send another Python to the wrong packages.
NOT_THEIRS = ("PYTHONPATH", "PYTHONHOME", "__PYVENV_LAUNCHER__")

WHERE = "https://github.com/bytedance/LatentSync"
_WHAT = {"repo": "folder", "python": "Python", "checkpoint": "file of weights"}


def options(settings: Settings, chosen: bool = True) -> dict[str, Any]:
    """What a face is moved with: the defaults, and over them what the setting says."""
    written = settings.extra(SLOT) if chosen else None
    kept = dict(DEFAULTS)
    if written:
        try:
            given = json.loads(written)
        except ValueError as error:
            raise LipSyncError(f"{SETTING} is not JSON: {error}") from None
        if not isinstance(given, dict):
            raise LipSyncError(f'{SETTING} should be a JSON object, such as {{"repo": "~/LatentSync"}}')
        unknown = sorted(set(given) - set(DEFAULTS))
        if unknown:
            raise LipSyncError(f"{SETTING} has {', '.join(unknown)}, which the latentsync adapter does not know; "
                               f"it knows: {', '.join(DEFAULTS)}")
        kept.update(given)
    for name in ("repo", "python", "checkpoint"):
        if kept[name] is not None and (not isinstance(kept[name], str) or not kept[name].strip()):
            raise LipSyncError(f"the {name} in {SETTING} is the path of a {_WHAT[name]}: not {kept[name]!r}")
    config, steps, guidance, seed = kept["config"], kept["steps"], kept["guidance"], kept["seed"]
    if not isinstance(config, str) or not config.strip():
        raise LipSyncError(f"the config in {SETTING} is the path of a file in the checkout: not {config!r}")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MOST_STEPS:
        raise LipSyncError(f"the steps in {SETTING} is a whole number from 1 to {MOST_STEPS}: not {steps!r}")
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not 1 <= guidance <= MOST_GUIDANCE:
        raise LipSyncError(f"the guidance in {SETTING} is a number from 1 to {MOST_GUIDANCE:g}: not {guidance!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LipSyncError(f"the seed in {SETTING} is a whole number, 0 or more: not {seed!r}")
    if not isinstance(kept["deepcache"], bool):
        raise LipSyncError(f"the deepcache in {SETTING} is true or false: not {kept['deepcache']!r}")
    return kept


class LatentSync:
    """LatentSync, run from a checkout of it by that checkout's own Python."""

    name = "latentsync"

    def __init__(self, model: str | None = None, *, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.from_env()
        # Nothing built in fills the slot, so with none chosen the settings are whoever's is asked for.
        chosen = self._settings.adapter(SLOT) in (None, self.name)
        self.options = options(self._settings, chosen)
        self.fetched_from = model or (self._settings.model(SLOT) if chosen else None) or DEFAULT_MODEL
        held = self.options["checkpoint"]
        #: What is printed beside what was made: the weights that made it.
        self.model = Path(held).name if held else self.fetched_from

    # The checkout.

    def checkout(self) -> Path:
        """The LatentSync checkout this runs, seen to be one."""
        named = self.options["repo"]
        if not named:
            raise LipSyncError(f"the latentsync adapter runs LatentSync from a checkout of it: git clone {WHERE}, "
                               f'and say where it is, as in {SETTING}=\'{{"repo": "~/LatentSync"}}\'')
        repo = Path(named).expanduser()
        missing = [one for one in HAS if not (repo / one).is_file()]
        if missing:
            raise LipSyncError(f"{repo} is not a checkout of LatentSync: it has no {missing[0]}")
        return repo.resolve()

    def python(self, repo: Path) -> Path:
        """The Python that has what LatentSync needs: the one named, or the checkout's own."""
        named = self.options["python"]
        if named:
            python = Path(named).expanduser()
            if not python.is_file():
                raise LipSyncError(f"the python in {SETTING} is not there: {python}")
            return python
        for one in PYTHONS:
            if (repo / one).is_file():
                return repo / one
        raise LipSyncError("the latentsync adapter runs LatentSync with a Python that has what LatentSync needs, "
                           f"and found none in {repo}: make one there as .venv, or name one, as in "
                           f'{SETTING}=\'{{"repo": "{self.options["repo"]}", "python": "/path/to/python"}}\'')

    def config(self, repo: Path) -> str:
        named = self.options["config"]
        if not (repo / named).is_file():
            raise LipSyncError(f"the config in {SETTING} is not in the checkout: {repo / named}")
        return str(named)

    def weights(self) -> Path | None:
        named = self.options["checkpoint"]
        if not named:
            return None
        held = Path(named).expanduser()
        if not held.is_file():
            raise LipSyncError(f"the checkpoint in {SETTING} is not there: {held}")
        return held.resolve()

    def faces(self, repo: Path) -> Path:
        """Where the face finder keeps its models: where the checkout has them, or with the engine's models."""
        own = repo / "checkpoints" / "auxiliary"
        return own if own.is_dir() else self._settings.models_dir / "latentsync" / "auxiliary"

    # Moving a mouth.

    def sync(self, video: Path, speech: Path, target: Path) -> Path:
        video, speech, target = Path(video), Path(speech), Path(target)
        repo = self.checkout()
        python, config, weights = self.python(repo), self.config(repo), self.weights()
        with tempfile.TemporaryDirectory(prefix="latentsync-") as folder:
            work = Path(folder)
            for one in LINKED:
                try:
                    (work / one).symlink_to(repo / one, target_is_directory=True)
                except OSError as error:
                    raise LipSyncError(f"the checkout's {one} could not be linked into {work}: {error}") from None
            shown = self._handed(video, speech, work)
            command = [
                str(python), str(RUNNER), "--repo", str(repo), "--model", self.fetched_from, "--config", config,
                "--faces", str(self.faces(repo)), "--video", shown, "--speech", "speech.wav", "--out", "synced.mp4",
                "--report", "report.json", "--device", self._settings.device,
                "--steps", str(self.options["steps"]), "--guidance", str(self.options["guidance"]),
                "--seed", str(self.options["seed"]),
            ]
            if weights is not None:
                command += ["--unet", str(weights)]
            if self.options["deepcache"]:
                command.append("--deepcache")
            environment = {name: value for name, value in os.environ.items() if name not in NOT_THEIRS}
            log = work / "log.txt"
            with gpu.alone(self._settings), log.open("wb") as printed:
                try:
                    done = subprocess.run(command, cwd=work, env=environment, stdin=subprocess.DEVNULL,
                                          stdout=printed, stderr=subprocess.STDOUT, check=False)
                except OSError as error:
                    raise LipSyncError(f"{python} could not be run: {error}") from None
            told = self._told(work / "report.json")
            made = work / "synced.mp4"
            if not told.get("ok") or done.returncode != 0 or not made.is_file():
                raise LipSyncError(self._wrong(told, done.returncode, log, video, python))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(made), target)
        return target

    def _handed(self, video: Path, speech: Path, work: Path) -> str:
        """Put the video and the speech where LatentSync is run, under names it can hand to a shell."""
        said = work / "speech.wav"
        try:
            ffmpeg.run(ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", speech, "-vn", "-ac", "1",
                       "-ar", str(RATE), "-c:a", "pcm_s16le", said)
        except ffmpeg.FfmpegError as error:
            raise LipSyncError(f"the speech in {speech.name} could not be read: {error}") from None
        heard = ffmpeg.probe(said)
        seconds = float(heard.duration_s or 0) if heard is not None else 0.0
        if not seconds:
            raise LipSyncError(f"{speech.name} holds no sound")
        suffix = video.suffix.lower() if re.fullmatch(r"\.[A-Za-z0-9]+", video.suffix) else ".mp4"
        shown = work / f"video{suffix}"
        try:
            # LatentSync holds every frame it is handed in memory, so it is handed no more than the speech needs.
            ffmpeg.run(ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-map", "0:v:0",
                       "-c", "copy", "-an", "-t", f"{seconds + PAST:.3f}", shown)
        except ffmpeg.FfmpegError:
            shutil.copyfile(video, shown)
        return shown.name

    @staticmethod
    def _told(report: Path) -> dict[str, Any]:
        try:
            told = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return told if isinstance(told, dict) else {}

    def _wrong(self, told: dict[str, Any], status: int, log: Path, video: Path, python: Path) -> str:
        """What went wrong, in words a person can act on."""
        kind, error = told.get("kind"), str(told.get("error") or "").strip()
        if kind == "no_face":
            return (f"LatentSync found no face in a frame of {video.name}: it needs a face in every frame, large "
                    "enough to see and turned towards the camera")
        if kind == "missing":
            return f"{error}. The Python that was run is {python}"
        if kind == "memory":
            return (f"LatentSync ran out of memory on {video.name}: it holds every frame at once, so a shorter clip "
                    f"or a smaller picture is the way round it ({error})")
        if error:
            return f"LatentSync could not move the mouth in {video.name}: {error}"
        try:
            lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        except OSError:
            lines = []
        tail = "\n".join(f"  {line}" for line in lines[-SHOWN:])
        return (f"LatentSync stopped with status {status} and said nothing of why"
                + (f". The last of what it printed:\n{tail}" if tail else ""))


def adapters() -> list[Adapter]:
    """What this package adds to the engine."""
    return [Adapter(
        slot=SLOT, name=LatentSync.name, factory=lazy("media_engine_latentsync:LatentSync"), license=LICENSE,
        summary="A face made to say a recording, with a LatentSync checkout on this machine; its face models are "
                "for non-commercial research",
    )]
