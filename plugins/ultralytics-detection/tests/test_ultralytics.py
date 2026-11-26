"""The Ultralytics detector: what it hands the engine, and the real library driven end to end.

No trained model is downloaded here. The real library is run on a network
built from its description and never trained, which finds nothing true but
goes through every step a trained one does.
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Ultralytics keeps a settings file. Keep it out of the home folder of whoever runs the tests.
_CONFIG = tempfile.TemporaryDirectory()
os.environ.setdefault("YOLO_CONFIG_DIR", _CONFIG.name)

import constructive_media_ultralytics as plugin  # noqa: E402
from constructive_media import adapters, detection  # noqa: E402
from constructive_media.adapters import Registry  # noqa: E402
from constructive_media.detection import Detection, Detector  # noqa: E402
from constructive_media.settings import Settings  # noqa: E402


def settings_in(folder, **changes):
    home = Path(folder)
    return Settings(home=home, work_dir=home / "work", models_dir=home / "models", **changes)


class Column:
    """A column of numbers that answers `tolist`, as a tensor does."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def result(found, width=640, height=360, names=None):
    """What Ultralytics answers for one image: `found` is (corners, kind, confidence) for each box."""
    return SimpleNamespace(
        orig_shape=(height, width),
        names=names or {0: "person", 16: "dog"},
        boxes=SimpleNamespace(
            xyxy=Column([box for box, _, _ in found]),
            cls=Column([float(kind) for _, kind, _ in found]),
            conf=Column([score for _, _, score in found]),
        ),
    )


class Network:
    """Stands in for a loaded model: answers each batch from what it was told to find."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def predict(self, images, **how):
        self.asked.append((list(images), how))
        return [self.answers[Path(image).name] for image in images]


class InAFolder(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.here = Path(folder.name)
        self.settings = settings_in(self.here)

    def detector(self, answers, **how):
        made = plugin.UltralyticsDetector(settings=self.settings, **how)
        made._network = Network(answers)
        return made


class Plugged(unittest.TestCase):
    def test_it_fills_the_detection_slot_under_its_own_license(self):
        (one,) = plugin.adapters()
        self.assertEqual((one.slot, one.name, one.license), ("detection", "ultralytics", "AGPL-3.0-only"))
        self.assertIsInstance(one.factory(), plugin.UltralyticsDetector)
        self.assertIsInstance(plugin.UltralyticsDetector(), Detector)

    def test_the_engine_finds_it_once_installed(self):
        found = adapters.registry().describe("detection", "ultralytics")
        self.assertEqual((found.license, found.origin), ("AGPL-3.0-only", "constructive-media-ultralytics"))
        self.assertIsInstance(detection.detector("ultralytics"), plugin.UltralyticsDetector)

    def test_it_never_becomes_the_default(self):
        self.assertEqual(adapters.registry().describe("detection").name, "onnx")

    def test_the_engine_would_not_take_it_as_a_built_in(self):
        (one,) = plugin.adapters()
        with self.assertRaises(adapters.LicenseError):
            Registry().register(one)


class Detecting(InAFolder):
    def test_boxes_come_back_as_shares_of_the_frame(self):
        detector = self.detector({"a.png": result([((64, 36, 320, 180), 0, 0.9), ((320, 180, 640, 360), 16, 0.5)])})
        ((person, dog),) = detector.detect([self.here / "a.png"])
        self.assertEqual(person, Detection("person", 0.9, 0.3, 0.3, 0.4, 0.4))
        self.assertEqual((dog.label, dog.confidence), ("dog", 0.5))
        self.assertEqual((dog.cx, dog.cy, dog.w, dog.h), (0.75, 0.75, 0.5, 0.5))

    def test_a_box_that_runs_off_the_frame_is_held_to_it(self):
        detector = self.detector({"a.png": result([((-20, -10, 320, 400), 0, 0.8)])})
        ((person,),) = detector.detect([self.here / "a.png"])
        self.assertEqual((person.cx, person.cy, person.w, person.h), (0.25, 0.5, 0.5, 1.0))

    def test_every_image_is_answered_in_the_order_given(self):
        answers = {f"{n}.png": result([((0, 0, 64, 36), 0, 0.9)] * (n % 3), width=64 * (n + 1)) for n in range(40)}
        detector = self.detector(answers)
        found = detector.detect([self.here / f"{n}.png" for n in range(40)], confidence=0.6)
        self.assertEqual([len(one) for one in found], [n % 3 for n in range(40)])
        self.assertAlmostEqual(found[4][0].w, 1 / 5)
        asked = detector._network.asked
        self.assertEqual([len(images) for images, _ in asked], [16, 16, 8])
        self.assertTrue(all(how == {"conf": 0.6, "device": "auto", "verbose": False} for _, how in asked))
        self.assertEqual(detector.detect([]), [])

    def test_an_image_with_nothing_in_it_is_an_empty_list(self):
        detector = self.detector({"a.png": result([]), "b.png": result([((0, 0, 64, 36), 0, 0.9)])})
        self.assertEqual([len(one) for one in detector.detect([self.here / "a.png", self.here / "b.png"])], [0, 1])


class Choosing(InAFolder):
    def test_the_model_comes_from_the_argument_then_the_settings_then_the_default(self):
        self.assertEqual(plugin.UltralyticsDetector(settings=self.settings).model, "yolo11n")
        chosen = settings_in(self.here, models={"detection": "yolo11s"})
        self.assertEqual(plugin.UltralyticsDetector(settings=chosen).model, "yolo11s")
        self.assertEqual(plugin.UltralyticsDetector("yolo11m", settings=chosen).model, "yolo11m")
        self.assertEqual(plugin.UltralyticsDetector(settings=chosen, device="cpu").device, "cpu")

    def test_a_model_known_by_name_is_kept_in_the_engine_s_models_folder(self):
        folder = self.here / "models" / "ultralytics"
        self.assertEqual(plugin.UltralyticsDetector(settings=self.settings).weights, folder / "yolo11n.pt")
        self.assertEqual(plugin.UltralyticsDetector("yolo11s.pt", settings=self.settings).weights, folder / "yolo11s.pt")

    def test_a_model_of_your_own_is_used_where_it_is(self):
        mine = self.here / "mine" / "best.pt"
        self.assertEqual(plugin.UltralyticsDetector(str(mine), settings=self.settings).weights, mine)
        self.assertEqual(plugin.UltralyticsDetector("yolo11n.yaml", settings=self.settings).weights, Path("yolo11n.yaml"))

    def test_nothing_is_loaded_until_something_is_asked(self):
        with mock.patch.dict("sys.modules", {"ultralytics": None, "torch": None}):
            made = plugin.UltralyticsDetector(settings=self.settings)
            self.assertIsNone(made._network)
            with self.assertRaises(ImportError):
                made.detect([self.here / "a.png"])


class TheRealLibrary(InAFolder):
    """Ultralytics itself, on a network that was never trained. Nothing is downloaded."""

    def test_a_network_is_built_run_and_answered_for_each_image(self):
        import numpy as np
        from PIL import Image

        noise = np.random.default_rng(3).integers(0, 255, (180, 320, 3), dtype=np.uint8)
        Image.fromarray(noise).save(self.here / "noise.png")
        Image.new("RGB", (320, 180), (40, 90, 160)).save(self.here / "blue.png")
        detector = plugin.UltralyticsDetector("yolo11n.yaml", device="cpu", settings=self.settings)
        found = detector.detect([self.here / "noise.png", self.here / "blue.png"], confidence=0.0001)
        self.assertEqual(len(found), 2)
        self.assertGreater(len(found[0]), 0)             # a network never trained guesses, and guesses freely
        for one in found[0] + found[1]:
            self.assertIsInstance(one.label, str)
            for share in (one.cx, one.cy, one.w, one.h):
                self.assertTrue(0.0 <= share <= 1.0)
            self.assertTrue(0.0001 <= one.confidence <= 1.0)
        self.assertEqual(list((self.here / "models").glob("**/*")), [])       # and nothing was fetched


if __name__ == "__main__":
    unittest.main()
