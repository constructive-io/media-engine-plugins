"""The pedalboard effects backend, run for real on sound made here."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pedalboard.io import AudioFile

import constructive_media_pedalboard as plugin
from constructive_media import adapters, effects
from constructive_media.adapters import Registry
from constructive_media.effects import REVERB_PRESETS, EffectBackend, EffectError, Effects

RATE = 48_000


def tone(seconds=0.5, loud=0.5, channels=1):
    """A sine that fades out over its last twentieth of a second, so its end is clean."""
    t = np.arange(int(seconds * RATE)) / RATE
    one = loud * np.sin(2 * np.pi * 440 * t) * np.minimum(1.0, (seconds - t) * 20)
    return np.tile(one.astype(np.float32), (channels, 1))


def click(seconds=0.25):
    """One sample at full scale, then silence."""
    sound = np.zeros((1, int(seconds * RATE)), dtype=np.float32)
    sound[0, 0] = 1.0
    return sound


def level(sound):
    return float(np.sqrt((sound.astype(np.float64) ** 2).mean()))


class InAFolder(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.here = Path(folder.name)
        self.backend = plugin.PedalboardEffects()

    def write(self, sound, name="dry.wav", depth=16):
        path = self.here / name
        with AudioFile(str(path), "w", RATE, sound.shape[0], bit_depth=depth) as out:
            out.write(sound)
        return path

    def read(self, path):
        with AudioFile(str(path)) as audio:
            return audio.read(audio.frames), str(audio.file_dtype)

    def through(self, sound, asked, depth=16):
        return self.read(self.backend.apply(self.write(sound, depth=depth), self.here / "wet.wav", asked))[0]


class Plugged(unittest.TestCase):
    def test_it_fills_the_effects_slot_under_its_own_license(self):
        (one,) = plugin.adapters()
        self.assertEqual((one.slot, one.name, one.license), ("effects", "pedalboard", "GPL-3.0-only"))
        self.assertIsInstance(one.factory(), plugin.PedalboardEffects)
        self.assertIsInstance(plugin.PedalboardEffects(), EffectBackend)

    def test_the_engine_finds_it_once_installed(self):
        found = adapters.registry().describe("effects", "pedalboard")
        self.assertEqual((found.license, found.origin), ("GPL-3.0-only", "constructive-media-pedalboard"))
        self.assertIsInstance(effects.backend("pedalboard"), plugin.PedalboardEffects)

    def test_it_never_becomes_the_default(self):
        self.assertEqual(adapters.registry().describe("effects").name, "ffmpeg")

    def test_the_engine_would_not_take_it_as_a_built_in(self):
        (one,) = plugin.adapters()
        with self.assertRaises(adapters.LicenseError):
            Registry().register(one)

    def test_it_offers_every_reverb_the_format_names(self):
        self.assertEqual(sorted(plugin.SPACES), sorted(REVERB_PRESETS))


class Applying(InAFolder):
    def test_nothing_asked_is_a_copy(self):
        dry = self.write(tone())
        wet = self.backend.apply(dry, self.here / "wet.wav", Effects())
        self.assertEqual(wet.read_bytes(), dry.read_bytes())

    def test_gain_is_exact(self):
        out = self.through(tone(), Effects(gain=-6))
        self.assertEqual(out.shape, (1, RATE // 2))
        self.assertAlmostEqual(level(out) / level(tone()), 10 ** (-6 / 20), places=3)

    def test_a_reverb_with_nothing_wet_leaves_the_sound_as_it_was(self):
        out = self.through(tone(), Effects(reverb="hall", reverb_mix=0.0))
        self.assertEqual(out.shape[1], RATE // 2)
        self.assertAlmostEqual(level(out) / level(tone()), 1.0, places=2)

    def test_a_reverb_rings_on_after_the_sound_ends(self):
        lengths = {}
        for name in REVERB_PRESETS:
            with self.subTest(reverb=name):
                out = self.through(tone(), Effects(reverb=name))
                lengths[name] = out.shape[1] / RATE
                after = out[:, int(0.55 * RATE):int(0.8 * RATE)]
                self.assertGreater(level(after), 0.001)
                # Rung out by the end: nothing is cut off mid-ring.
                self.assertLess(float(np.abs(out[:, -480:]).max()), 0.002)
                self.assertLessEqual(lengths[name], 0.5 + plugin.SPACES[name].rings + 0.01)
        self.assertGreater(lengths["cathedral"], lengths["hall"])
        self.assertGreater(lengths["hall"], lengths["room"])

    def test_the_dry_sound_stays_as_loud_as_the_mix_leaves_it(self):
        out = self.through(click(), Effects(reverb="room", reverb_mix=0.25))
        self.assertAlmostEqual(float(out[0, 0]), 0.75, places=2)

    def test_more_mix_is_more_room(self):
        little = self.through(tone(), Effects(reverb="hall", reverb_mix=0.1))
        much = self.through(tone(), Effects(reverb="hall", reverb_mix=0.6))
        after = slice(int(0.55 * RATE), int(0.8 * RATE))
        self.assertGreater(level(much[:, after]), 3 * level(little[:, after]))

    def test_a_delay_repeats_the_sound_each_quieter_than_the_last(self):
        out = self.through(click(), Effects(delay=0.2, delay_feedback=0.5, delay_mix=0.4))[0]
        taps = [round(float(out[int(n * 0.2 * RATE)]), 3) for n in range(4)]
        self.assertEqual(taps, [0.6, 0.4, 0.2, 0.1])
        self.assertAlmostEqual(float(np.abs(out[100:int(0.19 * RATE)]).max()), 0.0, places=4)

    def test_a_delay_is_left_to_ring_out(self):
        asked = Effects(delay=0.2, delay_feedback=0.5, delay_mix=0.4)
        out = self.through(click(), asked)[0]
        self.assertAlmostEqual(len(out) / RATE, 0.25 + plugin.ring(asked), places=1)
        self.assertLess(float(np.abs(out[-int(0.19 * RATE):]).max()), 0.002)

    def test_compression_brings_the_loud_down_and_leaves_the_quiet(self):
        loud = self.through(tone(loud=0.9), Effects(compress=True))
        quiet = self.through(tone(loud=0.05), Effects(compress=True))
        self.assertLess(level(loud) / level(tone(loud=0.9)), 0.5)
        self.assertAlmostEqual(level(quiet) / level(tone(loud=0.05)), 1.0, places=2)

    def test_an_unknown_reverb_says_what_there_is(self):
        with self.assertRaises(EffectError) as raised:
            self.through(tone(), Effects(reverb="cavern"))
        self.assertIn("room, hall, cathedral, plate", str(raised.exception))

    def test_a_file_that_is_not_sound_says_which(self):
        junk = self.here / "junk.wav"
        junk.write_text("not sound")
        with self.assertRaises(RuntimeError) as raised:
            self.backend.apply(junk, self.here / "wet.wav", Effects(gain=-3))
        self.assertIn("junk.wav", str(raised.exception))


class Files(InAFolder):
    def test_floats_stay_floats_and_pass_full_scale_unclipped(self):
        dry = self.write(tone(loud=2.5), depth=32)
        out, kind = self.read(self.backend.apply(dry, self.here / "wet.wav", Effects(gain=-6)))
        self.assertEqual(kind, "float32")
        self.assertAlmostEqual(float(np.abs(out).max()), 2.5 * 10 ** (-6 / 20), places=2)

    def test_whole_numbers_keep_their_depth(self):
        for depth, kind in ((16, "int16"), (24, "int24")):
            with self.subTest(depth=depth):
                dry = self.write(tone(), depth=depth)
                self.assertEqual(self.read(self.backend.apply(dry, self.here / "wet.wav", Effects(gain=-6)))[1], kind)

    def test_a_file_may_be_changed_in_place(self):
        dry = self.write(tone())
        same = self.backend.apply(dry, dry, Effects(gain=-6))
        self.assertEqual(same, dry)
        self.assertAlmostEqual(level(self.read(dry)[0]) / level(tone()), 10 ** (-6 / 20), places=3)
        self.assertEqual([path.name for path in self.here.iterdir()], ["dry.wav"])

    def test_two_channels_stay_two_and_a_reverb_spreads_them(self):
        both = tone(channels=2)
        both[1] *= 0.0                                   # all of it on the left
        out = self.through(both, Effects(reverb="hall", reverb_mix=0.5))
        self.assertEqual(out.shape[0], 2)
        self.assertGreater(level(out[1]), 0.001)         # the room is on both sides
        self.assertEqual(self.through(tone(), Effects(reverb="hall")).shape[0], 1)

    def test_other_kinds_of_file_are_written_as_what_they_are_named(self):
        wet = self.backend.apply(self.write(tone()), self.here / "wet.flac", Effects(gain=-6))
        self.assertEqual(wet.read_bytes()[:4], b"fLaC")
        self.assertAlmostEqual(level(self.read(wet)[0]) / level(tone()), 10 ** (-6 / 20), places=3)


class Ringing(unittest.TestCase):
    def test_how_long_a_request_rings(self):
        self.assertEqual(plugin.ring(Effects()), 0.0)
        self.assertEqual(plugin.ring(Effects(compress=True, gain=-3)), 0.0)
        self.assertEqual(plugin.ring(Effects(reverb="cathedral")), 4.5)
        self.assertEqual(plugin.ring(Effects(delay=0.25, delay_mix=0.0)), 0.0)
        self.assertEqual(plugin.ring(Effects(delay=0.25, delay_feedback=0.0)), 0.25)
        self.assertAlmostEqual(plugin.ring(Effects(delay=0.2, delay_feedback=0.5)), 0.2 * 11)
        self.assertAlmostEqual(plugin.ring(Effects(reverb="room", delay=0.2, delay_feedback=0.5)), 0.9 + 2.2)
        self.assertEqual(plugin.ring(Effects(delay=5, delay_feedback=0.95)), plugin.LONGEST_RING)

    def test_the_presets_ring_as_long_as_they_say(self):
        """Each preset's `rings` covers the time its room takes to fall sixty decibels."""
        import pedalboard

        click = np.zeros((1, 12 * RATE), dtype=np.float32)
        click[0, 0] = 1.0
        for name, space in plugin.SPACES.items():
            with self.subTest(reverb=name):
                room = pedalboard.Reverb(room_size=space.size, damping=space.damping, width=space.width,
                                         wet_level=1.0, dry_level=0.0)
                wet = pedalboard.Pedalboard([room])(click, RATE)[0].astype(np.float64)
                left = np.cumsum(wet[::-1] ** 2)[::-1]
                fallen = float(np.argmax(left < left[0] * 1e-6)) / RATE
                self.assertLessEqual(fallen, space.rings)
                self.assertGreater(fallen, space.rings * 0.6)


class InAMix(InAFolder):
    """The engine's mixer, with this backend chosen by name."""

    def test_a_track_s_reverb_is_given_the_room_this_backend_asks_for(self):
        from constructive_media.audio import mix

        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("the mixer reads sound with the ffmpeg program")
        mixer = mix.Mixer(mix.tracks({"nave": {"reverb": "cathedral"}}), backend="pedalboard")
        mixer.place("nave", tone()[0], 0.0)
        done = mixer.mix()
        self.assertAlmostEqual(done.report["seconds"], 0.5 + 4.5, places=2)
        after = done.stems["nave"][int(1.0 * RATE):int(2.0 * RATE)]
        self.assertGreater(level(after), 0.001)
        # A mono clip on a stereo track: the room is wide, the two sides differ.
        self.assertGreater(level(after[:, 0] - after[:, 1]), 0.0005)


if __name__ == "__main__":
    unittest.main()
