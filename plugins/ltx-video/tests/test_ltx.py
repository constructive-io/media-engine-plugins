"""The LTX-Video adapter.

No model is run here, and none is fetched. Stand-ins for `torch` and
`diffusers` keep what they were asked and answer with frames of one colour,
and the pictures a piece starts from are made by ffmpeg. One test, last,
checks the shape of the real diffusers when it is installed.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import media_engine_ltx as plugin
from constructive_media import adapters, ffmpeg, generate
from constructive_media.adapters import Registry
from constructive_media.generate import GenerateError, Maker, Request
from constructive_media.settings import Settings

WORDS = "fog drifting over a made-up harbour"
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def settings_in(folder, **changes):
    home = Path(folder)
    return Settings(home=home, work_dir=home / "work", models_dir=home / "models", **changes)


def painted(path, *bands):
    """A picture of bands side by side, each `(colour, width, height)`."""
    inputs, names = [], ""
    for at, ((red, green, blue), width, height) in enumerate(bands):
        inputs += ["-f", "lavfi", "-i", f"color=c=0x{red:02x}{green:02x}{blue:02x}:s={width}x{height}"]
        names += f"[{at}]"
    stacked = ["-filter_complex", f"{names}hstack=inputs={len(bands)}"] if len(bands) > 1 else []
    ffmpeg.run(ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *inputs, *stacked,
               "-frames:v", "1", "-update", "1", path)
    return Path(path)


class Pipeline:
    """A stand-in for the model: it answers with frames of one colour, and keeps what it was asked."""

    def __init__(self, old=False):
        self.asked, self.device, self.quiet, self.fails, self.empty = [], None, None, None, False
        self.scheduler = types.SimpleNamespace(config=types.SimpleNamespace(use_dynamic_shifting=old))

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, **how):
        self.quiet = how

    def __call__(self, **asked):
        self.asked.append(asked)
        if self.fails:
            raise RuntimeError(self.fails)
        count = 0 if self.empty else asked["num_frames"]
        frames = np.full((count, asked["height"], asked["width"], 3), 0.5, dtype=np.float32)
        frames[..., 0] = 1.5       # past what a picture can hold: it is kept in range
        return types.SimpleNamespace(frames=[frames])


def stand_ins(have=("mps",), broken=None, old=False):
    """What the adapter uses of `torch` and `diffusers`."""
    loads = []

    class Generator:
        def __init__(self, device=None):
            self.device, self.seed = device, None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16, torch.Generator = "float32", "bfloat16", Generator
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: "mps" in have))
    torch.cuda = types.SimpleNamespace(is_available=lambda: "cuda" in have)

    def pipeline(model, **how):
        if broken:
            raise OSError(broken)
        loads.append((model, how))
        return Pipeline(old)

    diffusers = types.ModuleType("diffusers")
    diffusers.LTXConditionPipeline = types.SimpleNamespace(from_pretrained=pipeline)
    return {"torch": torch, "diffusers": diffusers, "transformers": types.ModuleType("transformers")}, loads


class WithStandIns(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name).resolve()
        self.target = self.folder / "work" / "harbour.mp4"
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.loads = self.install()

    def install(self, **how):
        modules, loads = stand_ins(**how)
        patch = mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)
        return loads

    def maker(self, extra=None, chosen=True, **changes):
        if extra is not None:
            changes["extras"] = {"video": extra if isinstance(extra, str) else json.dumps(extra)}
        if chosen:
            changes["adapters"] = {"video": "ltx"}
        return plugin.LtxVideo(settings=settings_in(self.folder / "home", **changes))

    def asked(self, maker):
        _, pipeline = maker.loaded()
        (asked,) = pipeline.asked
        return asked


class Plugged(unittest.TestCase):
    def test_it_fills_the_video_slot_and_says_the_model_has_terms_of_its_own(self):
        (one,) = plugin.adapters()
        self.assertEqual((one.slot, one.name, one.license), ("video", "ltx", "LicenseRef-Lightricks-LTX-Video"))
        self.assertIn("terms of its own", one.summary)
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsInstance(plugin.LtxVideo(settings=settings_in(folder)), Maker)

    def test_the_engine_finds_it_once_installed(self):
        found = adapters.registry().describe("video", "ltx")
        self.assertEqual((found.license, found.origin), ("LicenseRef-Lightricks-LTX-Video", "media-engine-ltx"))
        with mock.patch.dict(os.environ, {plugin.SETTING: ""}):
            self.assertIsInstance(generate.maker_for("video", "ltx"), plugin.LtxVideo)

    def test_it_is_not_used_until_it_is_chosen(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(adapters.registry().describe("video", settings=settings_in(folder)).name, "diffusers")

    def test_the_engine_would_not_take_it_as_a_built_in(self):
        (one,) = plugin.adapters()
        with self.assertRaises(adapters.LicenseError):
            Registry().register(one)

    def test_nothing_heavy_is_imported_until_a_piece_is_made(self):
        code = ("import sys, media_engine_ltx as plugin; plugin.LtxVideo(); "
                "print(sorted(name for name in sys.modules "
                "if name.split('.')[0] in ('torch', 'diffusers', 'transformers', 'numpy', 'PIL')))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")


class Options(WithStandIns):
    def test_what_it_runs_with_unless_told(self):
        maker = self.maker()
        self.assertEqual((maker.name, maker.model), ("ltx", "Lightricks/LTX-Video-0.9.5"))
        self.assertEqual(maker.options, {
            "steps": 25, "fps": 24, "guidance": 3.0, "rough": 12,
            "negative": "worst quality, inconsistent motion, blurry, jittery, distorted",
        })

    def test_the_setting_goes_over_the_defaults(self):
        maker = self.maker({"steps": 40, "rough": 0}, models={"video": "made-up/later-model"})
        self.assertEqual((maker.model, maker.options["steps"], maker.options["rough"], maker.options["fps"]),
                         ("made-up/later-model", 40, 0, 24))

    def test_settings_meant_for_another_video_adapter_are_left_alone(self):
        # With the engine's own adapter chosen and this one asked for by name, its settings are not this one's.
        maker = self.maker({"steps": 15, "fps": 16, "another": True}, chosen=False,
                           models={"video": "made-up/words-only-model"})
        self.assertEqual((maker.model, maker.options), ("Lightricks/LTX-Video-0.9.5", plugin.DEFAULTS))

    def test_a_model_handed_over_is_the_model(self):
        made = plugin.LtxVideo("made-up/handed-over", settings=settings_in(self.folder / "home"))
        self.assertEqual(made.model, "made-up/handed-over")

    def test_what_it_cannot_run_with_is_refused_before_anything_is_loaded(self):
        for extra, said in (("{steps", "is not JSON"), ("[1]", "should be a JSON object"),
                            ({"passes": 3}, "has passes, which the ltx adapter does not know"),
                            ({"rough": 40}, "whole number from 0"), ({"rough": -1}, "whole number from 0"),
                            ({"rough": 2.5}, "whole number from 0"), ({"rough": True}, "whole number from 0")):
            with self.subTest(extra=extra), self.assertRaisesRegex(GenerateError, said):
                self.maker(extra)
        self.assertEqual(self.loads, [])

    def test_frames_come_eight_at_a_time_after_the_first(self):
        self.assertEqual([plugin.frames_for(seconds, 24) for seconds in (0.1, 1, 2, 4, 5)], [9, 25, 49, 97, 121])

    def test_a_size_in_a_picture_s_shape(self):
        self.assertEqual(plugin.size_like(1344, 768), (768, 448))
        self.assertEqual(plugin.size_like(1080, 1920), (448, 768))
        self.assertEqual(plugin.size_like(500, 500), (576, 576))
        self.assertEqual(plugin.size_like(4000, 20), (8224, 32))      # never a side of nothing


class FromWords(WithStandIns):
    def test_a_piece_is_made_with_the_defaults_and_plays(self):
        maker = self.maker()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            made = maker.make(Request("video", WORDS), self.target)
        self.assertEqual((made, out.getvalue()), (self.target, ""))
        self.assertEqual(self.asked(maker), {
            "prompt": WORDS, "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "height": 480, "width": 704, "num_frames": 49, "frame_rate": 24, "num_inference_steps": 25,
            "guidance_scale": 3.0, "decode_timestep": 0.05, "decode_noise_scale": 0.025, "output_type": "np",
        })
        described = ffmpeg.probe(made)
        self.assertEqual((described.medium, described.width, described.height, round(described.fps)),
                         ("video", 704, 480, 24))
        self.assertAlmostEqual(described.duration_s, 49 / 24, delta=0.1)
        # Red was past full and is held there; the rest is the grey it was.
        red, green, blue = ffmpeg.frame(made, 0.5, size=(1, 1))
        self.assertGreater(red, 200)
        self.assertLess(abs(green - blue), 40)

    def test_the_model_is_loaded_once_at_half_precision_and_kept_quiet(self):
        maker = self.maker()
        maker.make(Request("video", WORDS), self.target)
        maker.make(Request("video", WORDS + " again"), self.folder / "work" / "again.mp4")
        self.assertEqual(self.loads, [("Lightricks/LTX-Video-0.9.5", {"torch_dtype": "bfloat16"})])
        _, pipeline = maker.loaded()
        self.assertEqual((pipeline.device, pipeline.quiet, len(pipeline.asked)), ("mps", {"disable": True}, 2))

    def test_on_a_processor_it_runs_at_full_precision(self):
        self.loads = self.install(have=())
        maker = self.maker()
        maker.make(Request("video", WORDS), self.target)
        self.assertEqual(self.loads, [("Lightricks/LTX-Video-0.9.5", {"torch_dtype": "float32"})])
        self.assertEqual(maker.loaded()[1].device, "cpu")

    def test_the_device_asked_for_is_the_one_used_or_refused(self):
        self.loads = self.install(have=("mps", "cuda"))
        maker = self.maker(device="cuda")
        maker.make(Request("video", WORDS), self.target)
        self.assertEqual(maker.loaded()[1].device, "cuda")
        self.loads = self.install(have=())
        with self.assertRaisesRegex(GenerateError, "cannot use 'mps'"):
            self.maker(device="mps").make(Request("video", WORDS), self.target)
        with self.assertRaisesRegex(GenerateError, "runs on one of mps, cuda, cpu, not on 'tpu'"):
            self.maker(device="tpu").make(Request("video", WORDS), self.target)

    def test_what_the_request_says_goes_over_the_setting(self):
        maker = self.maker({"steps": 40, "fps": 30, "guidance": 4, "negative": ""})
        maker.make(Request("video", WORDS, width=512, height=320, seconds=1, steps=8, seed=7), self.target)
        asked = self.asked(maker)
        generator = asked.pop("generator")
        self.assertEqual((generator.device, generator.seed), ("cpu", 7))
        self.assertEqual(asked, {
            "prompt": WORDS, "negative_prompt": None, "height": 320, "width": 512, "num_frames": 33,
            "frame_rate": 30, "num_inference_steps": 8, "guidance_scale": 4.0, "decode_timestep": 0.05,
            "decode_noise_scale": 0.025, "output_type": "np",
        })
        self.assertEqual(round(ffmpeg.probe(self.target).fps), 30)

    def test_what_cannot_be_made_is_refused_before_a_model_is_loaded(self):
        maker = self.maker()
        with self.assertRaisesRegex(GenerateError, "makes video, not pictures"):
            maker.check(Request("image", WORDS))
        with self.assertRaisesRegex(GenerateError, "700x480 cannot be made: each side is a multiple of 32, such "
                                                   "as 704x480"):
            maker.make(Request("video", WORDS, width=700, height=480), self.target)
        with self.assertRaisesRegex(GenerateError, "such as 32x480"):
            maker.check(Request("video", WORDS, width=10, height=480))
        self.assertEqual(self.loads, [])
        self.assertFalse(self.target.exists())


class FromAPicture(WithStandIns):
    def test_the_picture_is_the_first_frame_in_its_own_shape(self):
        picture = painted(self.folder / "wide.png", (GREEN, 1344, 768))
        maker = self.maker()
        made = maker.make(Request("video", WORDS, image=picture), self.target)
        asked = self.asked(maker)
        first = asked.pop("image")
        self.assertEqual((asked["width"], asked["height"], first.size, first.mode), (768, 448, (768, 448), "RGB"))
        red, green, blue = first.getpixel((384, 224))
        self.assertTrue(red < 40 and green > 200 and blue < 40, (red, green, blue))
        self.assertEqual((ffmpeg.probe(made).width, ffmpeg.probe(made).height), (768, 448))

    def test_a_size_asked_for_is_filled_and_the_picture_is_not_squashed(self):
        # Four times as wide as it is tall: red, then green over the middle half, then blue.
        picture = painted(self.folder / "bands.png", (RED, 256, 256), (GREEN, 512, 256), (BLUE, 256, 256))
        maker = self.maker({"rough": 0})
        maker.make(Request("video", WORDS, image=picture, width=256, height=256), self.target)
        first = self.asked(maker)["image"]
        self.assertEqual(first.size, (256, 256))
        # Squashed, the sides would be red and blue. Cut from the middle, it is all green.
        for at in ((4, 128), (128, 128), (251, 128)):
            red, green, blue = first.getpixel(at)
            self.assertTrue(red < 40 and green > 200 and blue < 40, (at, red, green, blue))

    def test_it_is_roughened_unless_told_not_to(self):
        # A picture with detail in it: flat colour comes through any amount of roughening the same.
        picture = self.folder / "detail.png"
        ffmpeg.run(ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                   "-i", "testsrc2=s=256x256", "-frames:v", "1", str(picture))
        clean = np.frombuffer(plugin.first_frame(picture, 256, 256, 0), dtype=np.uint8).astype(int)
        some = np.frombuffer(plugin.first_frame(picture, 256, 256, 12), dtype=np.uint8).astype(int)
        more = np.frombuffer(plugin.first_frame(picture, 256, 256, 31), dtype=np.uint8).astype(int)
        self.assertEqual(len(clean), len(some), 256 * 256 * 3)
        self.assertLess(0, np.abs(some - clean).mean())
        self.assertLess(np.abs(some - clean).mean(), np.abs(more - clean).mean())
        self.assertLess(np.abs(more - clean).mean(), 40)       # rougher, and still the picture

    def test_a_picture_that_is_not_there_or_is_not_one_is_refused_before_a_model_is_loaded(self):
        maker = self.maker()
        with self.assertRaisesRegex(GenerateError, "not one that can be read: .*missing.png"):
            maker.check(Request("video", WORDS, image=self.folder / "missing.png"))
        words = self.folder / "words.png"
        words.write_text("not a picture")
        with self.assertRaisesRegex(GenerateError, "words.png"):
            maker.check(Request("video", WORDS, image=words))
        with self.assertRaisesRegex(GenerateError, "could not be read"):
            maker.make(Request("video", WORDS, image=words, width=256, height=256), self.target)
        self.assertEqual(self.loads, [])


class WhenItGoesWrong(WithStandIns):
    def test_a_model_that_will_not_load_is_said_in_a_line(self):
        self.loads = self.install(broken="no such repository")
        with self.assertRaisesRegex(GenerateError, "LTX-Video-0.9.5 could not be loaded: no such repository"):
            self.maker().make(Request("video", WORDS), self.target)

    def test_a_model_from_before_this_adapter_s_time_is_named_as_one(self):
        self.loads = self.install(old=True)
        maker = self.maker(models={"video": "made-up/an-early-one"})
        with self.assertRaisesRegex(GenerateError, "made-up/an-early-one is laid out as LTX-Video was before 0.9.5"):
            maker.make(Request("video", WORDS), self.target)
        self.assertFalse(self.target.exists())

    def test_what_the_model_raises_is_said_in_a_line(self):
        maker = self.maker()
        maker.loaded()[1].fails = "out of memory"
        with self.assertRaisesRegex(GenerateError, "could not make the piece: out of memory"):
            maker.make(Request("video", WORDS), self.target)

    def test_a_model_that_made_nothing_is_not_a_file(self):
        maker = self.maker()
        maker.loaded()[1].empty = True
        with self.assertRaisesRegex(GenerateError, "answered with nothing to show"):
            maker.make(Request("video", WORDS), self.target)
        self.assertFalse(self.target.exists())

    def test_without_the_libraries_it_says_what_to_install(self):
        with mock.patch.dict(sys.modules, {"diffusers": None}):
            with self.assertRaisesRegex(GenerateError, "pip install torch"):
                self.maker().make(Request("video", WORDS), self.target)

    def test_a_diffusers_too_old_to_have_the_model_says_the_same(self):
        with mock.patch.dict(sys.modules, {"diffusers": types.ModuleType("diffusers")}):
            with self.assertRaisesRegex(GenerateError, "a diffusers that has LTX-Video"):
                self.maker().make(Request("video", WORDS), self.target)


@unittest.skipUnless(importlib.util.find_spec("diffusers") and importlib.util.find_spec("torch"),
                     "diffusers and PyTorch are not installed")
class TheRealLibrary(unittest.TestCase):
    """The real diffusers takes everything this adapter hands it. No model is loaded and nothing fetched.

    In a program of its own, so that the stand-ins and the real thing never meet.
    """

    def test_it_takes_what_it_is_asked(self):
        code = (
            "import inspect, json, warnings\n"
            "warnings.simplefilter('ignore')\n"
            "from diffusers import LTXConditionPipeline\n"
            "print(json.dumps(sorted(inspect.signature(LTXConditionPipeline.__call__).parameters)))\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        takes = set(json.loads(done.stdout.strip().splitlines()[-1]))
        asked = {"prompt", "negative_prompt", "height", "width", "num_frames", "frame_rate", "num_inference_steps",
                 "guidance_scale", "decode_timestep", "decode_noise_scale", "output_type", "image", "generator"}
        self.assertEqual(asked - takes, set())


if __name__ == "__main__":
    unittest.main()
