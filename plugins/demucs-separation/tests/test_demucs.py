"""The Demucs adapter.

No trained model is run here, and none is fetched. Stand-ins for `torch`
and `demucs` keep what they were asked and answer with fixed shares of
what they were handed, and the recordings are tones made here. The last
test runs the real Demucs, when it is installed, on the small untrained
model its own tests use, which is built in memory.
"""

import importlib.util
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

import media_engine_demucs as plugin
from constructive_media import adapters, separation
from constructive_media.adapters import Registry
from constructive_media.audio import mix, pcm
from constructive_media.separation import SeparationError, Separator
from constructive_media.settings import Settings

RATE, THEIRS = 16_000, 44_100
#: How much of what it hears the stand-in calls each part. They add up to all of it.
SHARES = {"drums": 0.25, "bass": 0.125, "other": 0.125, "vocals": 0.5}
HALF, QUARTER, EIGHTH = -6.02, -12.04, -18.06


def settings_in(folder, **changes):
    home = Path(folder)
    return Settings(home=home, work_dir=home / "work", models_dir=home / "models", **changes)


def tone(hertz, level=0.2, seconds=1.0):
    return (level * np.sin(2 * np.pi * hertz * np.arange(int(seconds * RATE)) / RATE)).astype(np.float32)


class Tensor:
    """A stand-in for what the model answers with: numbers that say where they are kept."""

    def __init__(self, rows, where="mps"):
        self.rows, self.where = rows, where

    def detach(self):
        return self

    def cpu(self):
        return Tensor(self.rows, "cpu")

    def numpy(self):
        assert self.where == "cpu", "numbers were read off a graphics processor"
        return self.rows


def stand_ins(have=("mps",), broken=None, sources=tuple(SHARES), channels=2, fails=None):
    """What the adapter uses of `torch` and `demucs`."""
    built, heard = [], []

    class Model:
        def __init__(self, **how):
            if broken:
                raise RuntimeError(broken)
            built.append(how)
            self.samplerate, self.audio_channels = THEIRS, channels
            self.model = types.SimpleNamespace(sources=list(sources))

        def separate_tensor(self, wav, rate=None):
            heard.append((wav, rate))
            if fails:
                raise RuntimeError(fails)
            return wav, {name: Tensor(wav * SHARES.get(name, 0.0)) for name in sources}

    torch = types.ModuleType("torch")
    torch.from_numpy = lambda rows: rows
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: "mps" in have))
    torch.cuda = types.SimpleNamespace(is_available=lambda: "cuda" in have)
    api = types.ModuleType("demucs.api")
    api.Separator = Model
    demucs = types.ModuleType("demucs")
    demucs.api = api
    return {"torch": torch, "demucs": demucs, "demucs.api": api}, built, heard


class WithStandIns(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name).resolve()
        self.out = self.folder / "parts"
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.built, self.heard = self.install()

    def install(self, **how):
        modules, built, heard = stand_ins(**how)
        patch = mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)
        return built, heard

    def adapter(self, extra=None, chosen=True, **changes):
        if extra is not None:
            changes["extras"] = {"separation": extra if isinstance(extra, str) else json.dumps(extra)}
        if chosen:
            changes["adapters"] = {"separation": "demucs"}
        return plugin.DemucsSeparation(settings=settings_in(self.folder / "home", **changes))

    def recording(self, name="made-up-song.wav", *, two=True):
        sound = np.stack([tone(220), tone(3000, 0.1)], axis=1) if two else tone(220)
        return pcm.write(self.folder / name, sound, RATE, bits=16)

    def level(self, path):
        return mix.loudness(pcm.read(path, stereo=False))


class Plugged(unittest.TestCase):
    def test_it_fills_the_separation_slot_and_says_what_the_weights_are_for(self):
        (one,) = plugin.adapters()
        self.assertEqual((one.slot, one.name, one.license),
                         ("separation", "demucs", "LicenseRef-Demucs-weights-research-only"))
        self.assertIn("for research", one.summary)
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsInstance(plugin.DemucsSeparation(settings=settings_in(folder)), Separator)

    def test_the_engine_finds_it_once_installed(self):
        found = adapters.registry().describe("separation", "demucs")
        self.assertEqual((found.license, found.origin),
                         ("LicenseRef-Demucs-weights-research-only", "media-engine-demucs"))
        with mock.patch.dict(os.environ, {plugin.SETTING: ""}):
            self.assertIsInstance(separation.separator("demucs"), plugin.DemucsSeparation)

    def test_the_engine_would_not_take_it_as_a_built_in(self):
        (one,) = plugin.adapters()
        with self.assertRaises(adapters.LicenseError):
            Registry().register(one)

    def test_nothing_heavy_is_imported_until_a_recording_is_pulled_apart(self):
        code = ("import sys, media_engine_demucs as plugin; plugin.DemucsSeparation(); "
                "print(sorted(name for name in sys.modules "
                "if name.split('.')[0] in ('torch', 'demucs', 'torchaudio', 'numpy')))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")


class Options(WithStandIns):
    def test_what_it_runs_with_unless_told(self):
        made = self.adapter()
        self.assertEqual((made.name, made.model), ("demucs", "htdemucs"))
        self.assertEqual(made.options, {"parts": "two", "shifts": 0, "overlap": 0.25})

    def test_the_setting_goes_over_the_defaults(self):
        made = self.adapter({"parts": "all", "shifts": 2}, models={"separation": "made-up-model"})
        self.assertEqual((made.model, made.options), ("made-up-model", {"parts": "all", "shifts": 2, "overlap": 0.25}))

    def test_with_no_adapter_chosen_the_settings_are_its_own_when_it_is_asked_for(self):
        made = self.adapter({"shifts": 1}, chosen=False, models={"separation": "made-up-model"})
        self.assertEqual((made.model, made.options["shifts"]), ("made-up-model", 1))

    def test_settings_meant_for_another_separation_adapter_are_left_alone(self):
        made = self.adapter({"shifts": 4, "another": True}, chosen=False, adapters={"separation": "made-up-other"},
                            models={"separation": "made-up/other-model"})
        self.assertEqual((made.model, made.options), ("htdemucs", plugin.DEFAULTS))

    def test_a_model_handed_over_is_the_model(self):
        made = plugin.DemucsSeparation("made-up-handed-over", settings=settings_in(self.folder / "home"))
        self.assertEqual(made.model, "made-up-handed-over")

    def test_what_it_cannot_run_with_is_refused_before_anything_is_loaded(self):
        for extra, said in (("{parts", "is not JSON"), ("[1]", "should be a JSON object"),
                            ({"stems": 4}, "has stems, which the demucs adapter does not know"),
                            ({"parts": "four"}, "one of two, all: not 'four'"), ({"parts": 2}, "one of two, all"),
                            ({"shifts": -1}, "whole number from 0 to 10"), ({"shifts": 11}, "whole number from 0"),
                            ({"shifts": 1.5}, "whole number from 0"), ({"shifts": True}, "whole number from 0"),
                            ({"overlap": 1}, "from 0 to less than 1"), ({"overlap": -0.1}, "from 0 to less than 1"),
                            ({"overlap": "some"}, "from 0 to less than 1"), ({"overlap": False}, "from 0 to less")):
            with self.subTest(extra=extra), self.assertRaisesRegex(SeparationError, said):
                self.adapter(extra)
        self.assertEqual(self.built, [])


class Apart(WithStandIns):
    def test_a_recording_comes_back_as_the_voice_and_the_rest(self):
        made = self.adapter().separate(self.recording(), self.out)
        self.assertEqual(list(made), ["voice", "rest"])
        self.assertEqual({name: path.name for name, path in made.items()},
                         {"voice": "voice.wav", "rest": "rest.wav"})
        whole = self.level(self.folder / "made-up-song.wav")
        self.assertAlmostEqual(self.level(made["voice"]), whole + HALF, places=1)
        # The model's other parts are added up: a quarter and two eighths are the other half.
        self.assertAlmostEqual(self.level(made["rest"]), whole + HALF, places=1)

    def test_asked_for_every_part_each_keeps_the_model_s_name_for_it(self):
        made = self.adapter({"parts": "all"}).separate(self.recording(), self.out)
        self.assertEqual(list(made), ["voice", "drums", "bass", "other"])
        whole = self.level(self.folder / "made-up-song.wav")
        self.assertEqual([round(self.level(path) - whole, 1) for path in made.values()],
                         [round(HALF, 1), round(QUARTER, 1), round(EIGHTH, 1), round(EIGHTH, 1)])

    def test_the_model_hears_two_channels_at_its_own_rate_a_row_each(self):
        self.adapter().separate(self.recording(), self.out)
        ((wav, rate),) = self.heard
        self.assertEqual((rate, wav.shape, str(wav.dtype)), (THEIRS, (2, THEIRS), "float32"))
        self.assertTrue(wav.flags["C_CONTIGUOUS"])
        # The left is the low tone and the right the high one, as in the recording.
        self.assertAlmostEqual(mix.loudness(wav[0]), -17.0, places=0)
        self.assertAlmostEqual(mix.loudness(wav[1]), -23.0, places=0)

    def test_two_channels_come_back_as_two_with_each_where_it_was(self):
        made = self.adapter().separate(self.recording(), self.out)
        self.assertEqual(pcm.shape(made["voice"]), (2, THEIRS))
        voice = pcm.read(made["voice"])
        self.assertAlmostEqual(mix.loudness(voice[:, 0]), -17.0 + HALF, places=0)
        self.assertAlmostEqual(mix.loudness(voice[:, 1]), -23.0 + HALF, places=0)

    def test_one_channel_comes_back_as_one(self):
        made = self.adapter().separate(self.recording(two=False), self.out)
        self.assertEqual([pcm.shape(path) for path in made.values()], [(1, THEIRS), (1, THEIRS)])
        self.assertAlmostEqual(self.level(made["voice"]), -17.0 + HALF, places=0)

    def test_the_parts_run_as_long_as_the_recording(self):
        made = self.adapter().separate(self.recording(), self.out)
        self.assertEqual([pcm.seconds(path) for path in made.values()], [1.0, 1.0])

    def test_the_model_is_loaded_once_as_it_was_set_and_kept_quiet(self):
        made = self.adapter({"shifts": 3, "overlap": 0.5}, models={"separation": "made-up-model"})
        made.separate(self.recording(), self.out)
        made.separate(self.recording("made-up-other.wav"), self.folder / "others")
        self.assertEqual(self.built, [{"model": "made-up-model", "device": "mps", "shifts": 3, "overlap": 0.5,
                                       "progress": False}])
        self.assertEqual(len(self.heard), 2)
        self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "1")

    def test_the_device_asked_for_is_the_one_used_or_refused(self):
        self.built, self.heard = self.install(have=("cuda",))
        self.adapter().separate(self.recording(), self.out)
        self.adapter(device="cpu").separate(self.recording(), self.out)
        self.assertEqual([how["device"] for how in self.built], ["cuda", "cpu"])
        for device, said in (("mps", "cannot use 'mps' on this machine"), ("tpu", "one of mps, cuda, cpu, not on 'tpu'")):
            with self.subTest(device), self.assertRaisesRegex(SeparationError, said):
                self.adapter(device=device).separate(self.recording(), self.out)

    def test_through_the_engine_the_parts_are_written_where_they_were_asked_for(self):
        told = separation.apart(self.recording(), self.out, separator=self.adapter())
        self.assertEqual([one["part"] for one in told["parts"]], ["voice", "rest"])
        self.assertEqual((told["adapter"], told["model"], told["notes"]), ("demucs", "htdemucs", []))
        self.assertEqual(sorted(one.name for one in self.out.iterdir()), ["rest.wav", "voice.wav"])


class WhenItGoesWrong(WithStandIns):
    def test_a_model_that_will_not_load_is_said_in_a_line(self):
        self.built, self.heard = self.install(broken="made-up-model is neither a model nor a bag of them")
        with self.assertRaisesRegex(SeparationError, "made-up-model could not be loaded: made-up-model is neither"):
            plugin.DemucsSeparation("made-up-model", settings=settings_in(self.folder / "home")).separate(
                self.recording(), self.out)
        self.assertFalse(self.out.exists())

    def test_a_model_that_keeps_no_voice_apart_is_refused_with_what_it_does_keep(self):
        self.built, self.heard = self.install(sources=("drums", "bass"))
        with self.assertRaisesRegex(SeparationError, "keeps no voice apart: its parts are drums, bass"):
            self.adapter().separate(self.recording(), self.out)
        self.assertEqual(self.heard, [])

    def test_a_model_that_keeps_nothing_but_the_voice_is_refused(self):
        self.built, self.heard = self.install(sources=("vocals",))
        with self.assertRaisesRegex(SeparationError, "keeps no voice apart: its parts are vocals"):
            self.adapter().separate(self.recording(), self.out)

    def test_a_model_that_hears_another_number_of_channels_is_refused(self):
        self.built, self.heard = self.install(channels=1)
        with self.assertRaisesRegex(SeparationError, "hears 1 channels"):
            self.adapter().separate(self.recording(), self.out)

    def test_what_the_model_raises_is_said_in_a_line(self):
        self.built, self.heard = self.install(fails="out of memory")
        with self.assertRaisesRegex(SeparationError, "could not pull made-up-song.wav apart: out of memory"):
            self.adapter().separate(self.recording(), self.out)
        self.assertFalse(self.out.exists())

    def test_without_the_libraries_it_says_what_to_install(self):
        with mock.patch.dict(sys.modules, {"demucs": None, "demucs.api": None}):
            with self.assertRaisesRegex(SeparationError, "pip install .demucs"):
                self.adapter().separate(self.recording(), self.out)


@unittest.skipUnless(importlib.util.find_spec("demucs") and importlib.util.find_spec("torch"),
                     "Demucs and PyTorch are not installed")
class TheRealLibrary(unittest.TestCase):
    """The real Demucs, on the small model its own tests use: it is not trained, and nothing is fetched for it.

    What comes out is not a voice. That the adapter and the library fit each
    other is what is seen here: what it is handed, what it answers with, and
    the files that come of it. In a program of its own, so that the stand-ins
    and the real thing never meet.
    """

    def test_a_recording_goes_through_it_and_comes_back_in_parts(self):
        code = (
            "import json, sys, warnings\n"
            "warnings.simplefilter('ignore')\n"
            "from pathlib import Path\n"
            "import media_engine_demucs as plugin\n"
            "from constructive_media import separation\n"
            "from constructive_media.audio import pcm\n"
            "from constructive_media.settings import Settings\n"
            "home = Path(sys.argv[1])\n"
            "settings = Settings(home=home / 'home', work_dir=home / 'work', models_dir=home / 'models',\n"
            "                    device='cpu', extras={'separation': sys.argv[3]})\n"
            "does = plugin.DemucsSeparation('demucs_unittest', settings=settings)\n"
            "told = separation.apart(sys.argv[2], home / 'parts', separator=does)\n"
            "told['shapes'] = [pcm.shape(one['file']) for one in told['parts']]\n"
            "print(json.dumps(told))\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            song = pcm.write(folder / "made-up-song.wav", np.stack([tone(220, seconds=2.0), tone(3000, 0.1, 2.0)],
                                                                   axis=1), RATE, bits=16)
            alone = pcm.write(folder / "made-up-voice.wav", tone(220, seconds=2.0), RATE, bits=16)
            for recording, extra, parts, channels in ((song, "{}", ["voice", "rest"], 2),
                                                      (alone, '{"parts": "all", "shifts": 1}',
                                                       ["voice", "drums", "bass", "other"], 1)):
                with self.subTest(recording.name):
                    done = subprocess.run([sys.executable, "-c", code, str(folder / recording.stem), str(recording),
                                           extra], capture_output=True, text=True,
                                          env={**os.environ, "HF_HUB_OFFLINE": "1"})
                    self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                    told = json.loads(done.stdout.strip().splitlines()[-1])
                    self.assertEqual([one["part"] for one in told["parts"]], parts)
                    self.assertEqual((told["adapter"], told["model"]), ("demucs", "demucs_unittest"))
                    self.assertEqual(told["shapes"], [[channels, THEIRS]] * len(parts))
                    self.assertEqual([one["seconds"] for one in told["parts"]], [2.0] * len(parts))


if __name__ == "__main__":
    unittest.main()
