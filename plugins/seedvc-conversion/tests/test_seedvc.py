"""The Seed-VC voice conversion adapter.

No model is run here, and none is fetched. A stand-in for the `seed_vc`
package answers as the real one does: sound in and out in the same wrapping,
its models loaded afresh for every call, and its own chatter on standard
output. What it "converts" is the clip's sound, for as long as the recording
lasts, so that what the adapter handed it and did with the answer can be told
apart. One test, last, checks the real package's shape when it is installed.
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
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

import media_engine_seedvc as plugin
from constructive_media import adapters, conversion
from constructive_media.adapters import Registry
from constructive_media.audio import pcm
from constructive_media.conversion import ConversionError, Converter
from constructive_media.settings import Settings

RATE = plugin.RATE


def settings_in(folder, **changes):
    home = Path(folder)
    return Settings(home=home, work_dir=home / "work", models_dir=home / "models", **changes)


def tone(hertz, seconds, rate=48_000, loud=0.5):
    return (loud * np.sin(2 * np.pi * hertz * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


def pitch(samples, rate):
    """The loudest pitch in a sound, in hertz."""
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[int(spectrum.argmax())])


class Sound:
    """The package's own wrapping for sound."""

    def __init__(self, samples, mel_chunks, duration, samples_count, sample_rate, metadata):
        self.samples, self.sample_rate, self.duration, self.samples_count = samples, sample_rate, duration, samples_count


class Package:
    """A stand-in for `seed_vc`, as modules to put where the real ones would be."""

    def __init__(self, answers="wrapped", wrong=None):
        self.answers, self.wrong = answers, wrong
        self.loads, self.asked, self.fetched = 0, [], []
        self.api = types.ModuleType("seed_vc.api")
        self.api.load_models_realtime = self.load
        self.api.inference = self.inference
        self.loader = types.ModuleType("seed_vc.inference_realtime")
        self.loader.load_custom_model_from_hf = lambda *_, **__: self.fail("the package fetched to its own folder")
        sound = types.ModuleType("seed_vc.Models.audio")
        sound.AudioData = Sound
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.hf_hub_download = self.download
        self.modules = {
            "seed_vc": types.ModuleType("seed_vc"), "seed_vc.api": self.api, "seed_vc.inference_realtime": self.loader,
            "seed_vc.Models": types.ModuleType("seed_vc.Models"), "seed_vc.Models.audio": sound,
            "huggingface_hub": self.hub, "huggingface_hub.constants": types.ModuleType("huggingface_hub.constants"),
        }

    def fail(self, why):
        raise AssertionError(why)

    def download(self, repo_id, filename, **more):
        self.fetched.append((repo_id, filename, more))
        return f"/kept/{repo_id}/{filename}"

    def load(self, args):
        self.loads += 1
        return ("models", self.loads)

    def inference(self, source, target, **how):
        print("RTF: 0.25")
        # The real one asks for its models by the name its module holds, every time.
        models = self.api.load_models_realtime(types.SimpleNamespace(f0_condition=False, checkpoint=None, fp16=how["fp16"]))
        self.asked.append({"source": source, "target": target, "models": models, **how})
        if self.wrong is not None:
            raise self.wrong
        long = int(len(source.samples) * how["length_adjust"])
        made = np.resize(target.samples, long) if self.answers != "nothing" else target.samples[:0]
        if self.answers == "pair":
            return source.sample_rate, made.astype(np.float32) / 32768 * 4      # and past full scale
        return Sound(made, None, None, None, source.sample_rate, None)


class WithThePackage(unittest.TestCase):
    answers, wrong = "wrapped", None

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.here = Path(folder.name)
        self.package = Package(self.answers, self.wrong)
        patched = mock.patch.dict(sys.modules, self.package.modules)
        patched.start()
        self.addCleanup(patched.stop)
        self.take = pcm.write(self.here / "take.wav", tone(300, 2.0), 48_000)
        self.clip = pcm.write(self.here / "ada.wav", tone(900, 1.0), 48_000)

    def made(self, **extra):
        changes = {"extras": {"voice_conversion": json.dumps(extra)}} if extra else {}
        return plugin.SeedVC(settings=settings_in(self.here, **changes))

    def convert(self, converter=None, take=None, clip=None, name="out.wav"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            written = (converter or self.made()).convert(take or self.take, self.here / name, like=clip or self.clip)
        self.assertEqual(out.getvalue(), "")
        return written


class Plugged(unittest.TestCase):
    def test_it_fills_the_voice_conversion_slot_under_its_own_license(self):
        (one,) = plugin.adapters()
        self.assertEqual((one.slot, one.name, one.license), ("voice_conversion", "seedvc", "GPL-3.0-only"))
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsInstance(plugin.SeedVC(settings=settings_in(folder)), Converter)

    def test_the_engine_finds_it_once_installed(self):
        found = adapters.registry().describe("voice_conversion", "seedvc")
        self.assertEqual((found.license, found.origin), ("GPL-3.0-only", "media-engine-seedvc"))
        with mock.patch.dict(os.environ, {plugin.SETTING: ""}):
            self.assertIsInstance(conversion.converter("seedvc"), plugin.SeedVC)

    def test_it_is_not_used_until_it_is_chosen(self):
        with self.assertRaisesRegex(adapters.AdapterError, "Choose one of: .*seedvc"):
            adapters.registry().describe("voice_conversion")

    def test_the_engine_would_not_take_it_as_a_built_in(self):
        (one,) = plugin.adapters()
        with self.assertRaises(adapters.LicenseError):
            Registry().register(one)

    def test_nothing_of_the_package_is_imported_until_something_is_said(self):
        code = ("import sys, media_engine_seedvc as plugin; plugin.SeedVC(); "
                "print(sorted(name for name in sys.modules if name.split('.')[0] in ('seed_vc', 'torch', 'numpy')))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")


class Options(unittest.TestCase):
    def options(self, written=None):
        with tempfile.TemporaryDirectory() as folder:
            extras = {} if written is None else {"voice_conversion": written}
            return plugin.SeedVC(settings=settings_in(folder, extras=extras)).options

    def test_what_it_runs_with_unless_told(self):
        self.assertEqual(self.options(), {"steps": 10, "guidance": 0.7, "length": 1.0, "fp16": True})

    def test_the_setting_goes_over_the_defaults(self):
        self.assertEqual(self.options('{"steps": 4, "length": 2}'),
                         {"steps": 4, "guidance": 0.7, "length": 2.0, "fp16": True})

    def test_what_it_cannot_run_with_is_refused_before_anything_is_loaded(self):
        for written, said in (
            ("{steps", "is not JSON"),
            ("[4]", "should be a JSON object"),
            ('{"step": 4}', "says step, which the seedvc adapter does not take. It takes steps, guidance"),
            ('{"steps": 0}', "steps .* whole number from 1 to 200, not 0"),
            ('{"steps": 4.5}', "steps .* not 4.5"),
            ('{"steps": true}', "steps .* not True"),
            ('{"fp16": "yes"}', "fp16 .* true or false, not 'yes'"),
            ('{"guidance": 9}', "guidance .* from 0.0 to 5.0, not 9"),
            ('{"length": 0}', "length .* from 0.25 to 4.0, not 0"),
        ):
            with self.subTest(written=written), self.assertRaisesRegex(ConversionError, said):
                self.options(written)

    def test_it_runs_one_model(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(plugin.SeedVC(settings=settings_in(folder)).model, "Plachta/Seed-VC")
            plugin.SeedVC(settings=settings_in(folder, models={"voice_conversion": "Plachta/Seed-VC"}))
            with self.assertRaisesRegex(ConversionError, "runs one model, Plachta/Seed-VC, and was asked for another"):
                plugin.SeedVC(settings=settings_in(folder, models={"voice_conversion": "another"}))


class Converting(WithThePackage):
    def test_what_was_said_comes_back_as_long_and_in_the_clip_s_sound(self):
        written = self.convert()
        self.assertEqual(written, self.here / "out.wav")
        self.assertAlmostEqual(pcm.seconds(written), 2.0, places=2)
        self.assertEqual(pcm.read(written, stereo=False).shape, (2 * RATE,))
        self.assertAlmostEqual(pitch(pcm.read(written, stereo=False), RATE), 900, delta=2)

    def test_the_model_is_handed_one_channel_at_its_own_rate_in_whole_numbers(self):
        self.convert()
        (asked,) = self.package.asked
        for which, seconds, hertz in (("source", 2.0, 300), ("target", 1.0, 900)):
            handed = asked[which]
            self.assertEqual((handed.sample_rate, handed.samples.dtype, handed.samples.shape),
                             (RATE, np.int16, (int(seconds * RATE),)))
            self.assertAlmostEqual(pitch(handed.samples.astype(np.float32), RATE), hertz, delta=2)
            self.assertAlmostEqual(int(np.abs(handed.samples).max()), 0.5 * 32767, delta=400)
            self.assertEqual((handed.samples_count, handed.duration), (int(seconds * RATE), seconds))

    def test_it_is_run_with_what_the_setting_says(self):
        self.convert()
        self.convert(self.made(steps=4, guidance=0.3, length=1.5, fp16=False), name="slow.wav")
        first, second = ({name: asked[name] for name in ("diffusion_steps", "inference_cfg_rate", "length_adjust", "fp16")}
                         for asked in self.package.asked)
        self.assertEqual(first, {"diffusion_steps": 10, "inference_cfg_rate": 0.7, "length_adjust": 1.0, "fp16": True})
        self.assertEqual(second, {"diffusion_steps": 4, "inference_cfg_rate": 0.3, "length_adjust": 1.5, "fp16": False})
        self.assertAlmostEqual(pcm.seconds(self.here / "slow.wav"), 3.0, places=2)

    def test_no_more_of_a_clip_is_handed_over_than_the_model_hears(self):
        long = pcm.write(self.here / "long.wav", tone(900, 31.0, rate=16_000), 16_000)
        self.convert(clip=long)
        self.assertEqual(len(self.package.asked[0]["target"].samples), 25 * RATE)

    def test_the_models_are_loaded_once_however_much_is_said(self):
        converter = self.made()
        for name in ("one.wav", "two.wav", "three.wav"):
            self.convert(converter, name=name)
        self.assertEqual(self.package.loads, 1)
        self.assertEqual({asked["models"] for asked in self.package.asked}, {("models", 1)})
        # And by a second adapter in the same program, which the package's one set of models serves too.
        self.convert(self.made(), name="four.wav")
        self.assertEqual(self.package.loads, 1)

    def test_a_clip_is_read_once_and_again_when_it_changes(self):
        converter = self.made()
        self.convert(converter)
        self.convert(converter, name="again.wav")
        first, second = (asked["target"] for asked in self.package.asked)
        self.assertIs(first, second)
        pcm.write(self.clip, tone(700, 1.5), 48_000)
        self.convert(converter, name="changed.wav")
        self.assertAlmostEqual(pitch(pcm.read(self.here / "changed.wav", stereo=False), RATE), 700, delta=2)

    def test_only_so_many_clips_are_kept_ready(self):
        converter = self.made()
        for number in range(plugin.CLIPS_KEPT + 3):
            clip = pcm.write(self.here / f"clip-{number}.wav", tone(500 + 20 * number, 0.3), 48_000)
            self.convert(converter, clip=clip, name=f"as-{number}.wav")
        self.assertEqual(len(converter._clips), plugin.CLIPS_KEPT)
        self.assertIn(str((self.here / f"clip-{plugin.CLIPS_KEPT + 2}.wav").resolve()), {which[0] for which in converter._clips})

    def test_its_weights_are_fetched_to_where_models_are_kept_and_not_to_a_folder_of_its_own(self):
        self.convert()
        fetch = self.package.loader.load_custom_model_from_hf
        self.assertEqual(fetch("someone/model", "weights.pth", "how.yml"),
                         ("/kept/someone/model/weights.pth", "/kept/someone/model/how.yml"))
        self.assertEqual(fetch("someone/other", "weights.bin", config_filename=None), "/kept/someone/other/weights.bin")
        # Nothing says where: the place is Hugging Face's own, which is wherever you have it keep things.
        self.assertEqual({tuple(more) for _, _, more in self.package.fetched}, {()})

    def test_a_recording_with_nothing_in_it_is_not_handed_over(self):
        empty = pcm.write(self.here / "empty.wav", np.zeros(0, dtype=np.float32), 48_000)
        with self.assertRaisesRegex(ConversionError, "empty.wav has no sound in it"):
            self.convert(take=empty)
        with self.assertRaisesRegex(ConversionError, "empty.wav has no sound in it"):
            self.convert(clip=empty)
        self.assertEqual(self.package.asked, [])

    def test_through_the_engine(self):
        with mock.patch.dict(os.environ, {plugin.SETTING: '{"steps": 4}'}):
            adapters.registry()._instances.pop(("voice_conversion", "seedvc"), None)
            self.addCleanup(adapters.registry()._instances.pop, ("voice_conversion", "seedvc"), None)
            self.convert(conversion.converter("seedvc"))
        self.assertEqual(self.package.asked[0]["diffusion_steps"], 4)


class APair(WithThePackage):
    """The package has also answered with a rate and a sound of fractions."""

    answers = "pair"

    def test_it_is_taken_as_well_and_held_at_full_scale(self):
        heard = pcm.read(self.convert(), stereo=False)
        self.assertEqual(heard.shape, (2 * RATE,))
        self.assertAlmostEqual(float(np.abs(heard).max()), 1.0, places=3)


class Nothing(WithThePackage):
    answers = "nothing"

    def test_a_model_that_said_nothing_is_not_a_file(self):
        with self.assertRaisesRegex(ConversionError, "Seed-VC said nothing for take.wav"):
            self.convert()
        self.assertFalse((self.here / "out.wav").exists())


class Failing(WithThePackage):
    wrong = RuntimeError("out of memory")

    def test_what_the_model_raises_is_said_in_a_line(self):
        with self.assertRaisesRegex(ConversionError, "Seed-VC could not say take.wav again: out of memory"):
            self.convert()

    def test_and_the_next_sentence_can_still_be_tried(self):
        converter = self.made()
        with self.assertRaises(ConversionError):
            self.convert(converter)
        self.package.wrong = None
        self.assertTrue(self.convert(converter).is_file())


class Missing(WithThePackage):
    def test_without_the_package_it_says_what_to_install(self):
        with mock.patch.dict(sys.modules, {"seed_vc.api": None}), \
                self.assertRaisesRegex(ConversionError, "needs seed-vc: pip install seed-vc"):
            self.convert()

    def test_a_package_of_another_shape_is_said_to_be_one(self):
        del self.package.api.load_models_realtime
        with self.assertRaisesRegex(ConversionError, "no seed_vc.api.load_models_realtime: .* written for seed-vc 0.4"):
            self.convert()


@unittest.skipUnless(importlib.util.find_spec("seed_vc"), "seed-vc is not installed")
class TheRealPackage(unittest.TestCase):
    """What the real package does on being imported is undone, and it has the parts this adapter replaces.

    In a program of its own, because importing it cannot be taken back. No model is loaded and nothing fetched.
    """

    def test_it_is_left_behaving_as_a_library(self):
        code = (
            "import json, os, sys, warnings\n"
            "import media_engine_seedvc as plugin\n"
            "before = list(warnings.filters)\n"
            "api, sound = plugin.package()\n"
            "import huggingface_hub.constants as kept\n"
            "from seed_vc import inference_realtime as loader\n"
            "print(json.dumps({'cache': os.environ.get('HF_HUB_CACHE'), 'kept': kept.HF_HUB_CACHE,\n"
            "    'warnings': list(warnings.filters) == before, 'again': plugin.package()[0] is api,\n"
            "    'loads': api.load_models_realtime.__name__, 'fetches': loader.load_custom_model_from_hf.__name__,\n"
            "    'sound': sound.__name__, 'says': callable(api.inference)}))\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            environment = {**os.environ, "HF_HUB_CACHE": str(Path(folder) / "models")}
            done = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", code], cwd=folder,
                                  env=environment, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            told = json.loads(done.stdout.strip().splitlines()[-1])
            self.assertEqual(told, {
                "cache": str(Path(folder) / "models"), "kept": str(Path(folder) / "models"), "warnings": True,
                "again": True, "loads": "once", "fetches": "fetched", "sound": "AudioData", "says": True,
            })
            # It made itself no folder where the program was started.
            self.assertEqual(os.listdir(folder), [])


if __name__ == "__main__":
    warnings.simplefilter("default")
    unittest.main()
