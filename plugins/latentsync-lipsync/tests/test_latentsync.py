"""The LatentSync adapter, and the script it hands to LatentSync's own Python.

No model is run here and none is fetched, and there is no checkout of
LatentSync: a folder with the files the adapter looks for stands for one.
Where the adapter runs another Python it runs this one, on a script that
writes down what it was handed. The script that is really handed over is
run here too, in this process, against stand-ins for PyTorch and for
LatentSync that name an NVIDIA GPU wherever LatentSync's own code does, so
that what it stands in for is seen to be stood in for off one and left
alone on one. The video is a colour and the speech is a tone, made here.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import media_engine_latentsync as plugin
from constructive_media import adapters, ffmpeg, lipsync
from constructive_media.adapters import Registry
from constructive_media.lipsync import LipSyncError, Syncer
from constructive_media.settings import Settings
from media_engine_latentsync import runner

try:       # Before anything stands where a module would be: numpy cannot be loaded twice.
    import numpy as np
except ImportError:
    np = None

#: What stands where `runner.py` is when the adapter is tried: it writes down what it was handed, and does as told.
STAND_IN = '''
import json, os, shutil, sys, wave
asked = dict(zip(sys.argv[1::2], sys.argv[2::2]))
with wave.open(asked["--speech"], "rb") as held:
    speech = [held.getframerate(), held.getnchannels(), held.getsampwidth(), held.getnframes()]
seen = {
    "argv": sys.argv[1:], "asked": asked, "cwd": os.getcwd(), "here": sorted(os.listdir(".")),
    "links": {name: os.path.realpath(name) for name in ("configs", "latentsync")},
    "kept": {name: os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "STAND_IN_DOES")},
    "speech": speech,
}
with open(os.environ["STAND_IN_SEEN"], "w") as told:
    json.dump(seen, told)
does = os.environ.get("STAND_IN_DOES", "ok")
for line in range(20):
    print(f"line {line} of what LatentSync printed")
if does == "silent":
    sys.exit(3)
if does != "no_video":
    shutil.copyfile(asked["--video"], asked["--out"])
if does in ("ok", "no_video"):
    report = {"ok": True, "device": "cpu"}
else:
    report = {"ok": False, "kind": does, "error": "what LatentSync said"}
with open(asked["--report"], "w") as told:
    json.dump(report, told)
sys.exit(0 if report["ok"] else 1)
'''


def settings_in(folder, **changes):
    home = Path(folder)
    return Settings(home=home, work_dir=home / "work", models_dir=home / "models", **changes)


def checkout(folder):
    """A folder with what the adapter looks for in a checkout of LatentSync, and nothing of LatentSync in it."""
    repo = Path(folder) / "LatentSync"
    for one in (*plugin.HAS, plugin.DEFAULTS["config"]):
        (repo / one).parent.mkdir(parents=True, exist_ok=True)
        (repo / one).write_text("")
    return repo


def colour(path, seconds=3.0):
    ffmpeg.encode("-f", "lavfi", "-i", f"color=c=0x336699:s=160x90:r=25:d={seconds}", "-c:v", "{vcodec}",
                  "-pix_fmt", "yuv420p", path)
    return path


def tone(path, seconds=1.0, rate=24_000, channels=2):
    ffmpeg.run(ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
               "-i", f"sine=frequency=220:duration={seconds}", "-ar", str(rate), "-ac", str(channels), path)
    return path


def everything_in(folder):
    return sorted(str(one.relative_to(folder)) for one in Path(folder).rglob("*"))


class Options(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def made(self, written=None, **changes):
        if written is not None:
            changes["extras"] = {"lip_sync": written if isinstance(written, str) else json.dumps(written)}
        return plugin.LatentSync(settings=settings_in(self.folder, **changes))

    def test_the_engine_finds_it_under_its_own_license(self):
        (one,) = plugin.adapters()
        self.assertEqual((one.slot, one.name, one.license), ("lip_sync", "latentsync", plugin.LICENSE))
        self.assertTrue(plugin.LICENSE.startswith("LicenseRef-"))
        self.assertIn("non-commercial research", one.summary)
        # Nothing but a plugin may carry terms that are not the engine's own.
        with self.assertRaises(adapters.LicenseError):
            Registry().register(one)
        self.assertIsInstance(self.made(), Syncer)

    def test_nothing_of_latentsync_is_imported_by_the_adapter(self):
        code = ("import sys, media_engine_latentsync as plugin; plugin.LatentSync(); "
                "print(sorted(name for name in sys.modules if name.split('.')[0] in "
                "('torch', 'diffusers', 'latentsync', 'insightface', 'onnxruntime', 'numpy')))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")

    def test_the_runner_ships_beside_the_adapter(self):
        self.assertTrue(plugin.RUNNER.is_file())
        self.assertEqual(plugin.RUNNER.parent, Path(plugin.__file__).parent)

    def test_what_it_is_run_with_when_nothing_is_said(self):
        made = self.made()
        self.assertEqual(made.options, plugin.DEFAULTS)
        self.assertEqual((made.model, made.fetched_from), (plugin.DEFAULT_MODEL, plugin.DEFAULT_MODEL))
        self.assertEqual((made.options["steps"], made.options["guidance"], made.options["config"]),
                         (20, 1.5, "configs/unet/stage2.yaml"))

    def test_what_the_setting_says_goes_over_the_defaults(self):
        made = self.made({"repo": "~/LatentSync", "steps": 30, "guidance": 2, "seed": 7})
        self.assertEqual((made.options["repo"], made.options["steps"], made.options["guidance"],
                          made.options["seed"]), ("~/LatentSync", 30, 2, 7))
        self.assertEqual(made.options["config"], plugin.DEFAULTS["config"])

    def test_the_model_is_the_one_named(self):
        self.assertEqual(self.made(models={"lip_sync": "someone/LatentSync-fork"}).model, "someone/LatentSync-fork")
        named = plugin.LatentSync("another/one", settings=settings_in(self.folder))
        self.assertEqual((named.model, named.fetched_from), ("another/one", "another/one"))

    def test_weights_given_as_a_file_are_what_it_says_it_ran(self):
        made = self.made({"checkpoint": "/weights/tuned.pt"})
        self.assertEqual((made.model, made.fetched_from), ("tuned.pt", plugin.DEFAULT_MODEL))

    def test_settings_meant_for_another_adapter_are_left_alone(self):
        made = self.made({"endpoint": "https://example.test"}, adapters={"lip_sync": "another"},
                         models={"lip_sync": "another/model"})
        self.assertEqual((made.options, made.model), (plugin.DEFAULTS, plugin.DEFAULT_MODEL))

    def test_a_setting_it_does_not_know_is_refused(self):
        for written, words in (
            ("{not json", "is not JSON"),
            ("[1, 2]", "should be a JSON object"),
            ({"stpes": 30}, "has stpes, which the latentsync adapter does not know"),
            ({"repo": 3}, "the repo in .* is the path of a folder"),
            ({"python": ""}, "the python in .* is the path of a Python"),
            ({"checkpoint": ["a"]}, "the checkpoint in .* is the path of a file of weights"),
            ({"config": ""}, "the config in .* is the path of a file in the checkout"),
            ({"steps": 0}, "the steps in .* is a whole number from 1 to 100"),
            ({"steps": 101}, "the steps in .* is a whole number from 1 to 100"),
            ({"steps": 20.5}, "the steps in .* is a whole number from 1 to 100"),
            ({"steps": True}, "the steps in .* is a whole number from 1 to 100"),
            ({"guidance": 0.5}, "the guidance in .* is a number from 1 to 10"),
            ({"guidance": "high"}, "the guidance in .* is a number from 1 to 10"),
            ({"seed": -1}, "the seed in .* is a whole number, 0 or more"),
            ({"deepcache": "yes"}, "the deepcache in .* is true or false"),
            ({"deepcache": 1}, "the deepcache in .* is true or false"),
        ):
            with self.subTest(written=written), self.assertRaisesRegex(LipSyncError, words):
                self.made(written)


class Checkout(unittest.TestCase):
    """What is asked of the checkout before anything is run."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.repo = checkout(self.folder)

    def made(self, **written):
        return plugin.LatentSync(settings=settings_in(self.folder, extras={"lip_sync": json.dumps(written)}))

    def test_with_no_checkout_named_it_says_where_to_get_one(self):
        with self.assertRaisesRegex(LipSyncError, "git clone https://github.com/bytedance/LatentSync"):
            self.made().checkout()

    def test_a_folder_that_is_not_a_checkout_is_refused(self):
        (self.repo / plugin.HAS[0]).unlink()
        with self.assertRaisesRegex(LipSyncError, "is not a checkout of LatentSync: it has no latentsync/pipelines"):
            self.made(repo=str(self.repo)).checkout()
        with self.assertRaisesRegex(LipSyncError, "is not a checkout of LatentSync"):
            self.made(repo=str(self.folder / "nowhere")).checkout()

    def test_the_checkouts_own_python_is_found(self):
        made = self.made(repo=str(self.repo))
        with self.assertRaisesRegex(LipSyncError, "found none in .*make one there as .venv, or name one"):
            made.python(self.repo)
        (self.repo / "venv/bin").mkdir(parents=True)
        (self.repo / "venv/bin/python").write_text("")
        self.assertEqual(made.python(self.repo), self.repo / "venv/bin/python")
        (self.repo / ".venv/bin").mkdir(parents=True)
        (self.repo / ".venv/bin/python").write_text("")
        self.assertEqual(made.python(self.repo), self.repo / ".venv/bin/python")

    def test_a_python_that_is_named_is_the_one_used(self):
        self.assertEqual(self.made(repo=str(self.repo), python=sys.executable).python(self.repo),
                         Path(sys.executable))
        with self.assertRaisesRegex(LipSyncError, "the python in .* is not there"):
            self.made(repo=str(self.repo), python=str(self.folder / "no-python")).python(self.repo)

    def test_a_config_or_weights_that_are_not_there_are_refused(self):
        with self.assertRaisesRegex(LipSyncError, "the config in .* is not in the checkout"):
            self.made(repo=str(self.repo), config="configs/unet/other.yaml").config(self.repo)
        with self.assertRaisesRegex(LipSyncError, "the checkpoint in .* is not there"):
            self.made(repo=str(self.repo), checkpoint=str(self.folder / "no.pt")).weights()
        self.assertIsNone(self.made(repo=str(self.repo)).weights())

    def test_the_face_models_are_kept_with_the_checkout_when_it_has_them(self):
        made = self.made(repo=str(self.repo))
        self.assertEqual(made.faces(self.repo), self.folder / "models" / "latentsync" / "auxiliary")
        (self.repo / "checkpoints/auxiliary").mkdir(parents=True)
        self.assertEqual(made.faces(self.repo), self.repo / "checkpoints/auxiliary")


@unittest.skipUnless(shutil.which(ffmpeg.FFMPEG), "needs the ffmpeg program")
class Moving(unittest.TestCase):
    """A mouth moved, with this Python standing where LatentSync's would be."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.repo = checkout(self.folder)
        self.stand_in = self.folder / "stand_in.py"
        self.stand_in.write_text(STAND_IN)
        self.seen_at = self.folder / "seen.json"
        self.video = colour(self.folder / "face.mp4")
        self.speech = tone(self.folder / "line.wav")
        patches = (mock.patch.object(plugin, "RUNNER", self.stand_in),
                   mock.patch.dict(os.environ, {"STAND_IN_SEEN": str(self.seen_at), "PYTHONPATH": "/somewhere/else"}))
        for one in patches:
            one.start()
            self.addCleanup(one.stop)
        os.environ.pop("STAND_IN_DOES", None)

    def made(self, settings=None, **written):
        written = {"repo": str(self.repo), "python": sys.executable, **written}
        changes = {"extras": {"lip_sync": json.dumps(written)}, **(settings or {})}
        return plugin.LatentSync(settings=settings_in(self.folder, **changes))

    def seen(self):
        return json.loads(self.seen_at.read_text())

    def does(self, what):
        os.environ["STAND_IN_DOES"] = what

    def test_it_is_run_in_a_folder_of_its_own_with_the_checkout_linked_in(self):
        before = everything_in(self.repo)
        target = self.folder / "out" / "said.mp4"
        self.assertEqual(self.made().sync(self.video, self.speech, target), target)
        seen = self.seen()
        self.assertNotEqual(Path(seen["cwd"]).resolve(), self.repo.resolve())
        self.assertFalse(Path(seen["cwd"]).exists())                     # and it is gone when the run is over
        self.assertEqual({name: Path(where) for name, where in seen["links"].items()},
                         {"configs": (self.repo / "configs").resolve(),
                          "latentsync": (self.repo / "latentsync").resolve()})
        self.assertEqual(seen["here"], ["configs", "latentsync", "log.txt", "speech.wav", "video.mp4"])
        self.assertEqual(everything_in(self.repo), before)               # nothing was written into the checkout
        self.assertTrue(target.is_file())

    def test_what_latentsync_hands_to_a_shell_has_no_folder_in_its_name(self):
        odd = self.folder / "a clip (take 2); rm -rf.MOV"
        self.video.rename(odd)
        self.made().sync(odd, self.speech, self.folder / "said.mp4")
        asked = self.seen()["asked"]
        self.assertEqual((asked["--video"], asked["--speech"], asked["--out"], asked["--report"]),
                         ("video.mov", "speech.wav", "synced.mp4", "report.json"))

    def test_it_is_told_what_the_settings_say(self):
        weights = self.folder / "tuned.pt"
        weights.write_text("")
        self.made({"device": "cpu", "models": {"lip_sync": "someone/else"}}, steps=30, guidance=2.5, seed=7,
                  checkpoint=str(weights)).sync(self.video, self.speech, self.folder / "said.mp4")
        asked = self.seen()["asked"]
        self.assertEqual(Path(asked["--repo"]), self.repo.resolve())
        self.assertEqual((asked["--model"], asked["--config"], asked["--device"]),
                         ("someone/else", "configs/unet/stage2.yaml", "cpu"))
        self.assertEqual((asked["--steps"], asked["--guidance"], asked["--seed"]), ("30", "2.5", "7"))
        self.assertEqual(Path(asked["--unet"]), weights.resolve())
        self.assertEqual(Path(asked["--faces"]), self.folder / "models" / "latentsync" / "auxiliary")

    def test_with_nothing_said_it_is_left_to_the_machine_and_the_published_weights(self):
        self.made().sync(self.video, self.speech, self.folder / "said.mp4")
        asked = self.seen()["asked"]
        self.assertEqual((asked["--model"], asked["--device"], asked["--steps"], asked["--guidance"]),
                         (plugin.DEFAULT_MODEL, "auto", "20", "1.5"))
        self.assertNotIn("--unet", asked)
        self.assertNotIn("--deepcache", self.seen()["argv"])

    def test_deepcache_is_asked_for_when_the_settings_say_so(self):
        self.made(deepcache=True).sync(self.video, self.speech, self.folder / "said.mp4")
        self.assertEqual(self.seen()["argv"][-1], "--deepcache")

    def test_the_speech_is_handed_over_as_latentsync_hears_it(self):
        self.made().sync(self.video, self.speech, self.folder / "said.mp4")
        rate, channels, width, frames = self.seen()["speech"]
        self.assertEqual((rate, channels, width), (16_000, 1, 2))
        self.assertAlmostEqual(frames / rate, 1.0, delta=0.05)

    def test_no_more_of_the_video_is_handed_over_than_the_speech_needs(self):
        target = self.folder / "said.mp4"
        self.made().sync(self.video, self.speech, target)       # three seconds of video, one of speech
        self.assertAlmostEqual(ffmpeg.probe(target).duration_s, 1.0 + plugin.PAST, delta=0.2)

    def test_a_video_shorter_than_the_speech_is_handed_over_whole(self):
        target = self.folder / "said.mp4"
        self.made().sync(self.video, tone(self.folder / "long.wav", seconds=5.0), target)
        self.assertAlmostEqual(ffmpeg.probe(target).duration_s, 3.0, delta=0.2)

    def test_the_other_python_is_not_sent_to_this_ones_packages(self):
        self.made().sync(self.video, self.speech, self.folder / "said.mp4")
        self.assertEqual(self.seen()["kept"], {"PYTHONPATH": None, "PYTHONHOME": None, "STAND_IN_DOES": None})
        self.does("ok")
        self.made().sync(self.video, self.speech, self.folder / "again.mp4")
        self.assertEqual(self.seen()["kept"]["STAND_IN_DOES"], "ok")     # the rest of the environment is its own

    def test_a_frame_with_no_face_is_said_in_plain_words(self):
        self.does("no_face")
        with self.assertRaisesRegex(LipSyncError, "found no face in a frame of face.mp4: it needs a face in every"):
            self.made().sync(self.video, self.speech, self.folder / "said.mp4")
        self.assertFalse((self.folder / "said.mp4").exists())

    def test_a_python_without_what_latentsync_needs_is_named(self):
        self.does("missing")
        with self.assertRaises(LipSyncError) as caught:
            self.made().sync(self.video, self.speech, self.folder / "said.mp4")
        self.assertEqual(str(caught.exception), f"what LatentSync said. The Python that was run is {sys.executable}")

    def test_running_out_of_memory_says_what_to_do_about_it(self):
        self.does("memory")
        with self.assertRaisesRegex(LipSyncError, "ran out of memory on face.mp4: it holds every frame at once"):
            self.made().sync(self.video, self.speech, self.folder / "said.mp4")

    def test_anything_else_it_said_is_passed_on(self):
        self.does("failed")
        with self.assertRaisesRegex(LipSyncError, "could not move the mouth in face.mp4: what LatentSync said"):
            self.made().sync(self.video, self.speech, self.folder / "said.mp4")

    def test_when_it_stops_without_a_word_the_last_of_what_it_printed_is_shown(self):
        self.does("silent")
        with self.assertRaises(LipSyncError) as caught:
            self.made().sync(self.video, self.speech, self.folder / "said.mp4")
        said = str(caught.exception)
        self.assertIn("stopped with status 3 and said nothing of why", said)
        self.assertIn("line 19 of what LatentSync printed", said)
        self.assertIn(f"line {20 - plugin.SHOWN} of", said)
        self.assertNotIn(f"line {20 - plugin.SHOWN - 1} of", said)

    def test_saying_it_went_well_and_writing_nothing_is_a_failure(self):
        self.does("no_video")
        with self.assertRaisesRegex(LipSyncError, "stopped with status 0 and said nothing of why"):
            self.made().sync(self.video, self.speech, self.folder / "said.mp4")

    def test_a_python_that_cannot_be_run_is_said_to_be_one(self):
        broken = self.folder / "python"
        broken.write_text("not a program")
        with self.assertRaisesRegex(LipSyncError, "could not be run"):
            self.made(python=str(broken)).sync(self.video, self.speech, self.folder / "said.mp4")

    def test_a_recording_with_no_sound_in_it_is_refused_before_anything_is_run(self):
        with self.assertRaisesRegex(LipSyncError, "could not be read|holds no sound"):
            self.made().sync(self.video, self.video, self.folder / "said.mp4")
        self.assertFalse(self.seen_at.exists())

    def test_through_the_engine_the_speech_is_laid_under_what_came_back(self):
        target = self.folder / "said.mp4"
        told = lipsync.sync(self.video, self.speech, target, syncer=self.made())
        self.assertEqual((told["adapter"], told["model"], told["notes"]), ("latentsync", plugin.DEFAULT_MODEL, []))
        written = ffmpeg.probe(target)
        self.assertTrue(written.has_audio)
        self.assertAlmostEqual(written.duration_s, 1.0, delta=0.15)

    def test_the_real_script_says_what_a_python_is_missing(self):
        # This Python has no LatentSync in it, whatever else it has: the folder is not a checkout of anything.
        with mock.patch.object(plugin, "RUNNER", Path(runner.__file__)):
            with self.assertRaises(LipSyncError) as caught:
                self.made().sync(self.video, self.speech, self.folder / "said.mp4")
        self.assertIn("this Python does not have what LatentSync needs", str(caught.exception))
        self.assertIn(f"The Python that was run is {sys.executable}", str(caught.exception))


# The script, against stand-ins.


class Kind:
    """A stand-in for one of PyTorch's kinds of number."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


HALF, FULL = Kind("float16"), Kind("float32")


def stand_ins(have=("mps",), capability=8, decord=False, face=True, writes=True):
    """What the script uses of PyTorch, diffusers and LatentSync, and a record of what each was asked.

    The stand-in for LatentSync names an NVIDIA GPU where LatentSync does:
    its pipeline builds its face tools on "cuda" whatever it was moved to,
    its face finder asks for the CUDA provider and turns the device into a
    GPU's number, its face tools work at half precision unless told, and on
    the CPU it builds no face finder at all.
    """
    asked = types.SimpleNamespace(seed=None, moved=None, called=None, read_with=None, analysis=[], prepared=[],
                                  restorers=[], finders=[], listener=None, vae=None, unet=[], fetched=[],
                                  scheduler=None)

    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = HALF, FULL
    torch.cuda = types.SimpleNamespace(is_available=lambda: "cuda" in have,
                                       get_device_capability=lambda: (capability, 0))
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: "mps" in have))
    torch.from_numpy = lambda numbers: numbers

    accelerate = types.ModuleType("accelerate")
    accelerate.utils = types.ModuleType("accelerate.utils")
    accelerate.utils.set_seed = lambda seed: setattr(asked, "seed", seed)

    class AutoencoderKL:
        @classmethod
        def from_pretrained(cls, name, torch_dtype=None):
            asked.vae = (name, torch_dtype)
            return types.SimpleNamespace(config=types.SimpleNamespace(scaling_factor=None, shift_factor=None))

    class DDIMScheduler:
        @classmethod
        def from_pretrained(cls, folder):
            asked.scheduler = folder
            return "scheduler"

    diffusers = types.ModuleType("diffusers")
    diffusers.AutoencoderKL, diffusers.DDIMScheduler = AutoencoderKL, DDIMScheduler

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(cross_attention_dim=384),
        data=types.SimpleNamespace(num_frames=16, audio_feat_length=[2, 2], resolution=256,
                                   mask_image_path="latentsync/utils/mask.png"),
    )
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(load=lambda path: config, to_container=lambda part: {"a": "model"})

    class Unet:
        def to(self, dtype=None):
            asked.unet.append(dtype)
            return self

    class UNet3DConditionModel:
        @classmethod
        def from_pretrained(cls, model, path, device="cpu"):
            asked.unet.append((model, path, device))
            return Unet(), 0

    detector = types.ModuleType("latentsync.utils.face_detector")

    class FaceAnalysis:
        def __init__(self, **how):
            asked.analysis.append(how)

        def prepare(self, ctx_id=0, det_size=None):
            asked.prepared.append(ctx_id)

    def cuda_to_int(name):
        if not name.startswith("cuda"):
            raise ValueError(f"Device must be cuda, got {name}")
        return 0

    class FaceDetector:
        def __init__(self, device="cuda"):
            asked.finders.append(device)
            self.app = detector.FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"],
                                             root="checkpoints/auxiliary", providers=["CUDAExecutionProvider"])
            self.app.prepare(ctx_id=detector.cuda_to_int(device), det_size=(512, 512))

    detector.FaceAnalysis, detector.cuda_to_int, detector.FaceDetector = FaceAnalysis, cuda_to_int, FaceDetector

    class AlignRestore:
        def __init__(self, align_points=3, resolution=256, device="cpu", dtype=HALF):
            asked.restorers.append((resolution, device, dtype))

    affine = types.ModuleType("latentsync.utils.affine_transform")
    affine.AlignRestore = AlignRestore

    pipeline = types.ModuleType("latentsync.pipelines.lipsync_pipeline")

    class ImageProcessor:
        def __init__(self, resolution=512, device="cpu", mask_image=None):
            self.restorer = AlignRestore(resolution=resolution, device=device)
            self.face_detector = None if device == "cpu" else detector.FaceDetector(device=device)

    class LipsyncPipeline:
        def __init__(self, **parts):
            self.parts = parts

        def to(self, device):
            asked.moved = device
            return self

        def __call__(self, **how):
            asked.called = how
            asked.read_with = pipeline.read_audio
            self.image_processor = pipeline.ImageProcessor(how["height"], device="cuda", mask_image=None)
            if not face:
                raise RuntimeError("Face not detected")
            if writes:
                Path(how["video_out_path"]).write_bytes(b"a video")

    pipeline.ImageProcessor, pipeline.LipsyncPipeline = ImageProcessor, LipsyncPipeline
    pipeline.read_audio = "LatentSync's own reader"

    class Audio2Feature:
        def __init__(self, **how):
            asked.listener = how

    hub = types.ModuleType("huggingface_hub")

    def hf_hub_download(model, name):
        asked.fetched.append((model, name))
        return f"/kept/{model}/{name}"

    hub.hf_hub_download = hf_hub_download

    latentsync = types.ModuleType("latentsync")
    modules = {
        "torch": torch, "accelerate": accelerate, "accelerate.utils": accelerate.utils, "diffusers": diffusers,
        "omegaconf": omegaconf, "huggingface_hub": hub, "latentsync": latentsync,
        "latentsync.models": types.ModuleType("latentsync.models"),
        "latentsync.models.unet": types.ModuleType("latentsync.models.unet"),
        "latentsync.pipelines": types.ModuleType("latentsync.pipelines"),
        "latentsync.pipelines.lipsync_pipeline": pipeline,
        "latentsync.utils": types.ModuleType("latentsync.utils"),
        "latentsync.utils.face_detector": detector, "latentsync.utils.affine_transform": affine,
        "latentsync.whisper": types.ModuleType("latentsync.whisper"),
        "latentsync.whisper.audio2feature": types.ModuleType("latentsync.whisper.audio2feature"),
        # Nothing where a module would be is how Python is told there is none.
        "decord": types.ModuleType("decord") if decord else None,
    }
    modules["latentsync.models.unet"].UNet3DConditionModel = UNet3DConditionModel
    modules["latentsync.pipelines"].lipsync_pipeline = pipeline
    modules["latentsync.utils"].face_detector = detector
    modules["latentsync.whisper.audio2feature"].Audio2Feature = Audio2Feature
    return modules, asked, pipeline, detector


class Script(unittest.TestCase):
    """`runner.py`, which is handed to LatentSync's own Python."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.repo = self.folder / "LatentSync"
        self.repo.mkdir()
        path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), path))
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def asked(self, **changes):
        given = {"repo": str(self.repo), "model": "someone/LatentSync", "config": "configs/unet/stage2.yaml",
                 "faces": str(self.folder / "faces"), "unet": None, "video": "video.mp4", "speech": "speech.wav",
                 "out": str(self.folder / "synced.mp4"), "report": str(self.folder / "report.json"),
                 "device": "auto", "steps": 20, "guidance": 1.5, "seed": 1247, "deepcache": False, **changes}
        return argparse.Namespace(**given)

    def run_with(self, modules, **changes):
        with mock.patch.dict(sys.modules, modules):
            return runner.run(self.asked(**changes))

    # Off an NVIDIA GPU.

    def test_off_an_nvidia_gpu_everything_that_names_one_is_stood_in_for(self):
        modules, asked, pipeline, _ = stand_ins(have=("mps",))
        told = self.run_with(modules)
        self.assertEqual((told["ok"], told["device"], told["precision"], told["stood_in"]),
                         (True, "mps", "float32", True))
        self.assertEqual((asked.moved, asked.listener["device"]), ("mps", "mps"))
        self.assertEqual((asked.vae, asked.unet[-1], asked.called["weight_dtype"]),
                         (("stabilityai/sd-vae-ft-mse", FULL), FULL, FULL))
        # The pipeline asked for its face tools on "cuda", and was given them where the machine can run them.
        self.assertEqual(asked.finders, ["mps"])
        self.assertEqual(asked.restorers[-1], (256, "mps", FULL))
        self.assertEqual(asked.analysis[-1]["providers"], ["CPUExecutionProvider"])
        self.assertEqual(asked.prepared, [-1])
        self.assertIs(asked.read_with, runner.read_audio)

    def test_on_the_cpu_there_is_still_a_face_finder(self):
        modules, asked, _, _ = stand_ins(have=())
        told = self.run_with(modules)
        self.assertEqual((told["device"], told["precision"], asked.moved), ("cpu", "float32", "cpu"))
        self.assertEqual((asked.finders, asked.prepared), (["cpu"], [-1]))
        self.assertEqual(asked.restorers[-1], (256, "cpu", FULL))

    def test_a_device_can_be_asked_for_and_one_the_machine_lacks_is_refused(self):
        modules, asked, _, _ = stand_ins(have=("mps",))
        self.assertEqual(self.run_with(modules, device="cpu")["device"], "cpu")
        for device, words in (("cuda", "this PyTorch cannot use 'cuda' on this machine"),
                              ("gpu", "runs on one of cuda, mps, cpu, not on 'gpu'")):
            with self.subTest(device=device), self.assertRaisesRegex(runner.Refusal, words):
                self.run_with(stand_ins(have=("mps",))[0], device=device)

    # On one.

    def test_on_an_nvidia_gpu_latentsync_is_left_as_it_is(self):
        modules, asked, pipeline, detector = stand_ins(have=("cuda",), decord=True)
        processor, numbering = pipeline.ImageProcessor, detector.cuda_to_int
        told = self.run_with(modules)
        self.assertEqual((told["device"], told["precision"], told["stood_in"]), ("cuda", "float16", False))
        self.assertIs(pipeline.ImageProcessor, processor)
        self.assertIs(detector.cuda_to_int, numbering)
        self.assertEqual(asked.read_with, "LatentSync's own reader")
        self.assertEqual((asked.finders, asked.prepared), (["cuda"], [0]))
        self.assertEqual(asked.restorers, [(256, "cuda", HALF)])
        self.assertEqual(asked.analysis[-1]["providers"], ["CUDAExecutionProvider"])
        self.assertEqual((asked.vae[1], asked.called["weight_dtype"], asked.moved), (HALF, HALF, "cuda"))

    def test_an_older_nvidia_gpu_runs_at_full_precision_as_latentsync_has_it(self):
        modules, asked, _, _ = stand_ins(have=("cuda",), capability=7, decord=True)
        self.assertEqual(self.run_with(modules)["precision"], "float32")
        self.assertEqual(asked.vae[1], FULL)

    # Either way.

    def test_it_makes_the_calls_latentsyncs_own_script_makes(self):
        modules, asked, _, _ = stand_ins()
        told = self.run_with(modules, steps=30, guidance=2.0, seed=7)
        self.assertEqual((asked.scheduler, asked.seed), ("configs", 7))
        self.assertEqual(asked.unet[0], ({"a": "model"}, "/kept/someone/LatentSync/latentsync_unet.pt", "cpu"))
        self.assertEqual(asked.listener, {"model_path": "/kept/someone/LatentSync/whisper/tiny.pt", "device": "mps",
                                          "num_frames": 16, "audio_feat_length": [2, 2]})
        self.assertEqual(asked.called, {
            "video_path": "video.mp4", "audio_path": "speech.wav", "video_out_path": str(self.folder / "synced.mp4"),
            "num_frames": 16, "num_inference_steps": 30, "guidance_scale": 2.0, "weight_dtype": FULL,
            "width": 256, "height": 256, "mask_image_path": "latentsync/utils/mask.png", "temp_dir": "temp",
        })
        self.assertEqual((told["steps"], told["resolution"]), (30, 256))
        self.assertEqual(sys.path[0], str(self.repo))

    def test_deepcache_is_left_alone_unless_asked_for_and_set_as_latentsyncs_script_sets_it(self):
        modules, asked, _, _ = stand_ins()
        modules["DeepCache"] = None                 # not installed, and not missed
        self.assertFalse(self.run_with(modules)["deepcache"])

        done = []

        class DeepCacheSDHelper:
            def __init__(self, pipe=None):
                done.append(("made", type(pipe).__name__, asked.moved, asked.seed))

            def set_params(self, **how):
                done.append(("set", how))

            def enable(self):
                done.append(("enabled", asked.called))

        modules, asked, _, _ = stand_ins()
        modules["DeepCache"] = types.ModuleType("DeepCache")
        modules["DeepCache"].DeepCacheSDHelper = DeepCacheSDHelper
        self.assertTrue(self.run_with(modules, deepcache=True)["deepcache"])
        # On the pipeline once it is where it runs, before the seed is set and before a mouth is moved.
        self.assertEqual(done, [("made", "LipsyncPipeline", "mps", None),
                                ("set", {"cache_interval": 3, "cache_branch_id": 0}), ("enabled", None)])

    def test_deepcache_asked_of_a_python_without_it_is_refused(self):
        modules, asked, _, _ = stand_ins()
        modules["DeepCache"] = None
        with self.assertRaisesRegex(runner.Refusal, "pip install DeepCache") as caught:
            self.run_with(modules, deepcache=True)
        self.assertEqual(caught.exception.kind, "missing")
        self.assertIsNone(asked.called)

    def test_the_face_models_are_kept_where_it_was_told(self):
        modules, asked, _, _ = stand_ins()
        self.run_with(modules)
        self.assertEqual(asked.analysis[-1]["root"], str(self.folder / "faces"))
        self.assertTrue((self.folder / "faces").is_dir())
        self.assertEqual(asked.analysis[-1]["allowed_modules"], ["detection", "landmark_2d_106"])

    def test_weights_given_as_a_file_are_not_fetched(self):
        modules, asked, _, _ = stand_ins()
        self.run_with(modules, unet="/weights/tuned.pt")
        self.assertEqual(asked.unet[0][1], "/weights/tuned.pt")
        self.assertEqual(asked.fetched, [("someone/LatentSync", "whisper/tiny.pt")])

    def test_the_checkouts_own_speech_model_is_used_when_it_has_one(self):
        own = self.repo / "checkpoints" / "whisper" / "tiny.pt"
        own.parent.mkdir(parents=True)
        own.write_text("")
        modules, asked, _, _ = stand_ins()
        self.run_with(modules)
        self.assertEqual(asked.listener["model_path"], str(own))
        self.assertEqual(asked.fetched, [("someone/LatentSync", "latentsync_unet.pt")])

    def test_a_run_that_writes_nothing_is_not_called_a_success(self):
        modules, _, _, _ = stand_ins(writes=False)
        with self.assertRaisesRegex(runner.Refusal, "ran to its end and wrote no video"):
            self.run_with(modules)

    def test_what_was_there_before_is_not_taken_for_what_was_made(self):
        (self.folder / "synced.mp4").write_bytes(b"from before")
        modules, _, _, _ = stand_ins(writes=False)
        with self.assertRaises(runner.Refusal):
            self.run_with(modules)

    def test_a_python_without_latentsync_is_said_to_be_one(self):
        modules, _, _, _ = stand_ins()
        modules["latentsync.models.unet"] = None
        with self.assertRaises(runner.Refusal) as caught:
            self.run_with(modules)
        self.assertEqual(caught.exception.kind, "missing")
        self.assertIn("this Python does not have what LatentSync needs", str(caught.exception))

    # What it writes down.

    def arguments(self, **changes):
        asked = vars(self.asked(**changes))
        switches = [f"--{name}" for name, value in asked.items() if value is True]
        return [part for name, value in asked.items() if value is not None and not isinstance(value, bool)
                for part in (f"--{name}", str(value))] + switches

    def test_what_happened_is_written_where_it_was_asked(self):
        modules, _, _, _ = stand_ins()
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(runner.main(self.arguments()), 0)
        told = json.loads((self.folder / "report.json").read_text())
        self.assertEqual((told["ok"], told["device"], told["steps"]), (True, "mps", 20))
        self.assertGreaterEqual(told["loading"], 0)
        self.assertGreaterEqual(told["moving"], 0)

    def test_a_frame_with_no_face_is_written_down_as_that(self):
        modules, _, _, _ = stand_ins(face=False)
        with mock.patch.dict(sys.modules, modules), mock.patch.object(runner.traceback, "print_exc"):
            self.assertEqual(runner.main(self.arguments()), 1)
        told = json.loads((self.folder / "report.json").read_text())
        self.assertEqual((told["ok"], told["kind"]), (False, "no_face"))

    def test_what_went_wrong_is_sorted_into_kinds(self):
        self.assertEqual(runner.said(runner.Refusal("no", "weights")), {"ok": False, "kind": "weights", "error": "no"})
        self.assertEqual(runner.said(RuntimeError("Face not detected"))["kind"], "no_face")
        self.assertEqual(runner.said(MemoryError())["kind"], "memory")
        self.assertEqual(runner.said(RuntimeError("MPS backend out of memory (MPS allocated: 60 GB)"))["kind"],
                         "memory")
        self.assertEqual(runner.said(KeyError("frames")),
                         {"ok": False, "kind": "failed", "error": "KeyError: 'frames'"})

    # Its parts.

    def test_the_best_device_there_is_is_picked(self):
        self.assertEqual(runner.pick("auto", {"cuda": True, "mps": True, "cpu": True}), "cuda")
        self.assertEqual(runner.pick("auto", {"cuda": False, "mps": True, "cpu": True}), "mps")
        self.assertEqual(runner.pick("auto", {"cuda": False, "mps": False, "cpu": True}), "cpu")
        self.assertEqual(runner.pick("cpu", {"cuda": True, "cpu": True}), "cpu")

    def test_a_speech_model_is_had_for_each_width_latentsync_has(self):
        modules, asked, _, _ = stand_ins()
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(runner.listens_through(str(self.repo), "a/model", 768), "/kept/a/model/whisper/small.pt")
            with self.assertRaisesRegex(runner.Refusal, "listens 512 wide is not one LatentSync has"):
                runner.listens_through(str(self.repo), "a/model", 512)

    def test_weights_that_cannot_be_fetched_are_said_to_be_that(self):
        modules, _, _, _ = stand_ins()

        def fails(model, name):
            raise OSError("no network")

        modules["huggingface_hub"].hf_hub_download = fails
        with mock.patch.dict(sys.modules, modules), self.assertRaises(runner.Refusal) as caught:
            runner.fetched("a/model", "latentsync_unet.pt")
        self.assertEqual(caught.exception.kind, "weights")
        self.assertIn("latentsync_unet.pt of a/model could not be fetched: no network", str(caught.exception))
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}), self.assertRaises(runner.Refusal) as caught:
            runner.fetched("a/model", "latentsync_unet.pt")
        self.assertEqual(caught.exception.kind, "missing")

    def test_where_decord_is_not_installed_something_stands_where_it_would_be(self):
        with mock.patch.dict(sys.modules, {"decord": None}):
            self.assertTrue(runner.without_decord())
            import decord

            with self.assertRaisesRegex(RuntimeError, "decord is not installed"):
                decord.AudioReader("speech.wav")
        here = types.ModuleType("decord")
        with mock.patch.dict(sys.modules, {"decord": here}):
            self.assertFalse(runner.without_decord())
            self.assertIs(sys.modules["decord"], here)


@unittest.skipUnless(np is not None, "needs numpy")
class Reading(unittest.TestCase):
    """The speech, read in place of the reader LatentSync has no build of for some machines."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        torch = types.ModuleType("torch")
        torch.from_numpy = lambda numbers: numbers
        modules = mock.patch.dict(sys.modules, {"torch": torch})
        modules.start()
        self.addCleanup(modules.stop)

    def written(self, samples, rate=16_000, channels=1, width=2):
        path = self.folder / "speech.wav"
        with wave.open(str(path), "wb") as held:
            held.setnchannels(channels)
            held.setsampwidth(width)
            held.setframerate(rate)
            held.writeframes(np.asarray(samples, dtype="<i2" if width == 2 else "u1").tobytes())
        return str(path)

    def test_it_is_read_as_numbers_from_minus_one_to_one(self):
        read = runner.read_audio(self.written([0, 16384, -32768, 32767]))
        self.assertEqual(str(read.dtype), "float32")
        self.assertEqual([round(float(one), 4) for one in read], [0.0, 0.5, -1.0, 1.0])

    def test_two_channels_are_read_as_one(self):
        read = runner.read_audio(self.written([16384, 0, -16384, -16384], channels=2))
        self.assertEqual([round(float(one), 4) for one in read], [0.25, -0.5])

    def test_another_rate_is_brought_to_the_one_asked_for(self):
        librosa = types.ModuleType("librosa")
        librosa.resample = lambda samples, orig_sr, target_sr: np.repeat(samples, target_sr // orig_sr)
        with mock.patch.dict(sys.modules, {"librosa": librosa}):
            read = runner.read_audio(self.written([0, 16384], rate=8_000))
        self.assertEqual([round(float(one), 4) for one in read], [0.0, 0.0, 0.5, 0.5])

    def test_a_recording_that_is_not_what_was_handed_over_is_refused(self):
        with self.assertRaisesRegex(runner.Refusal, "is not the 16-bit recording this was handed"):
            runner.read_audio(self.written([0, 128, 255], width=1))


if __name__ == "__main__":
    unittest.main()
