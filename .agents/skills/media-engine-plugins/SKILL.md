---
name: media-engine-plugins
description: Plugins for Constructive Media — adapters for tools under copyleft licenses (GPL, AGPL), each a package of its own. Use when asked to "install a media engine plugin", "use pedalboard effects", "use Ultralytics YOLO for detection", "use Seed-VC", "convert a voice", "say a recording again in another voice", "install voice conversion", "switch the effects backend", "switch the detector", "write a plugin for the media engine", "add an adapter", or when a tool somebody wants is not MIT, Apache-2.0 or BSD licensed.
compatibility: Python 3.11+, with constructive-media installed in the same environment
metadata:
  author: constructive-io
  version: "0.2.0"
---

# Media engine plugins

[Constructive Media](https://github.com/constructive-io/media-engine) is
MIT licensed and contains only MIT, Apache-2.0 and BSD code. A tool under
another license is driven by a plugin from this repository: a separate package,
under that tool's license, which the engine finds once it is installed and
uses only when it is chosen.

## When to apply

- Installing a plugin, choosing it, or checking which adapter a slot is using.
- Writing an adapter for a tool whose license the engine cannot hold.
- Answering what installing a plugin does to the license of what is built.

For using the engine itself (setting it up, a `.conv` script to sound, footage
to a cut), use the `constructive-media` skill in the engine's repository.

## The plugins

| Plugin | Fills the slot | Chosen as | Tool | License |
| --- | --- | --- | --- | --- |
| [`pedalboard-effects`](../../../plugins/pedalboard-effects) | `effects` | `pedalboard` | pedalboard | GPL-3.0-only |
| [`ultralytics-detection`](../../../plugins/ultralytics-detection) | `detection` | `ultralytics` | Ultralytics YOLO | AGPL-3.0-only |
| [`seedvc-conversion`](../../../plugins/seedvc-conversion) | `voice_conversion` | `seedvc` | Seed-VC | GPL-3.0-only |

Each wraps the real library, which it installs as a dependency. Nothing is
reimplemented here.

## Install, choose, check

Into the environment the engine is installed in:

```sh
pip install ./plugins/pedalboard-effects
pip install ./plugins/ultralytics-detection
pip install ./plugins/seedvc-conversion      # Python 3.11 or 3.12 only

media adapters   # every slot, what can fill it, its license and where it came from
```

Installing changes nothing. A plugin is never the default; choose it by slot:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_EFFECTS=pedalboard
export CONSTRUCTIVE_MEDIA_ADAPTER_DETECTION=ultralytics
export CONSTRUCTIVE_MEDIA_MODEL_DETECTION=yolo11n    # a name Ultralytics publishes, or a model file
export CONSTRUCTIVE_MEDIA_ADAPTER_VOICE_CONVERSION=seedvc
```

`media adapters` marks the adapter in use with `*`. Unset the variable to go back
to the engine's own. The engine has none of its own for `voice_conversion`:
`media revoice` and `media talk --quick` work only with a plugin chosen.

## Voice conversion

```sh
media revoice take.wav out/take-as-ada.wav --voice ada      # a persona whose voice was cloned from a clip
media revoice take.wav out/take-as-ada.wav --like ada.flac  # or any recording of the voice
media talk --voice ada --quick af_heart                     # a quick voice, said again in ada's
```

- **It fetches about 2.8 GB of weights the first time**, into Hugging Face's
  usual cache. Say so before the first run, and do not call it hung.
- **`media revoice` prints how like the voice the result is**, beside how like
  it the recording was. About 0.8 is as near as a clone. One test voice came
  out 0.79 and another 0.38 to 0.53: report the number, and under about 0.6
  say that it leans towards the voice and is not it.
- **More `steps` are slower and were no nearer the voice.** Leave the default
  (10) unless somebody hears a reason; 4 for talking saves a fifth of a second
  a sentence. Options go in `CONSTRUCTIVE_MEDIA_EXTRA_VOICE_CONVERSION` as
  JSON, and are listed in the plugin's README.
- **It needs a NumPy older than 2**, so an environment on Python 3.13, or one
  whose other packages need NumPy 2, cannot hold it. Give it one of its own.
- **Two people's say-so.** The voice is somebody's and so is the recording
  that is said again. Do not convert into a voice without its owner's yes.

## Before recommending one

- **Say what the license means.** A program that includes a GPL part is covered
  by the GPL when it is passed on. The AGPL extends that to people who use it
  over a network. Tell the person before installing, and do not install a
  plugin as a side effect of something else.
- **Ultralytics may send usage statistics** unless that is turned off
  (`yolo settings sync=False`). The engine sends nothing.
- **The engine does not need pedalboard or Ultralytics.** Its own effects run
  through `ffmpeg` and its own detectors are Apache-2.0. Measure before
  claiming a plugin is better for the job at hand. Voice conversion is the one
  thing here the engine cannot do without a plugin.

## Writing a plugin

One folder under `plugins/`, one package, one license:

```
plugins/my-tool/
├── LICENSE              # the license of the tool it drives
├── README.md
├── pyproject.toml
├── media_engine_my_tool/__init__.py
└── tests/
```

`pyproject.toml` names the license and the entry point:

```toml
[project]
name = "media-engine-my-tool"
license = "GPL-3.0-only"
dependencies = ["constructive-media", "the-tool"]

[project.entry-points."constructive_media.adapters"]
my_tool = "media_engine_my_tool:adapters"
```

and the entry point returns the adapters:

```python
from constructive_media.adapters import Adapter, lazy

def adapters():
    return [
        Adapter(
            slot="effects",
            name="my-tool",
            factory=lazy("media_engine_my_tool:MyEffects"),
            license="GPL-3.0-only",     # the SPDX identifier, stated honestly
            summary="What it does, in a line",
        )
    ]
```

What the object must do is set by its slot, in the engine's
[docs/ADAPTERS.md](https://github.com/constructive-io/media-engine/blob/main/docs/ADAPTERS.md).
Rules that hold for every plugin:

- Import the tool's library only inside the adapter, never at the top of the
  module the entry point is in, so listing adapters costs nothing.
- Import the engine's interfaces; do not copy them here, and never copy code
  from here into the engine.
- No model weights, voices, audio, images or footage. Tests make their own and
  download nothing.

Test from the plugin's folder, with the engine and the plugin installed:

```sh
cd plugins/pedalboard-effects && python -m unittest discover -s tests -t .
```

## Related skills

| Skill | Where | For |
| --- | --- | --- |
| `constructive-media` | [media-engine](https://github.com/constructive-io/media-engine) | Setting up and using the engine |
| `conv` | [conversation-format](https://github.com/constructive-io/conversation-format) | Writing and checking the `.conv` scripts the engine renders |
