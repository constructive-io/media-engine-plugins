---
name: media-engine-plugins
description: Plugins for Constructive Media — adapters for tools under copyleft licenses (GPL, AGPL), each a package of its own. Use when asked to "install a media engine plugin", "use pedalboard effects", "use Ultralytics YOLO for detection", "switch the effects backend", "switch the detector", "write a plugin for the media engine", "add an adapter", or when a tool somebody wants is not MIT, Apache-2.0 or BSD licensed.
compatibility: Python 3.11+, with constructive-media installed in the same environment
metadata:
  author: constructive-io
  version: "0.1.0"
---

# Media engine plugins

[Constructive Media](https://github.com/constructive-io/constructive-media) is
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

Each wraps the real library, which it installs as a dependency. Nothing is
reimplemented here.

## Install, choose, check

Into the environment the engine is installed in:

```sh
pip install ./plugins/pedalboard-effects
pip install ./plugins/ultralytics-detection

me adapters      # every slot, what can fill it, its license and where it came from
```

Installing changes nothing. A plugin is never the default; choose it by slot:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_EFFECTS=pedalboard
export CONSTRUCTIVE_MEDIA_ADAPTER_DETECTION=ultralytics
export CONSTRUCTIVE_MEDIA_MODEL_DETECTION=yolo11n    # a name Ultralytics publishes, or a model file
```

`me adapters` marks the adapter in use with `*`. Unset the variable to go back
to the engine's own.

## Before recommending one

- **Say what the license means.** A program that includes a GPL part is covered
  by the GPL when it is passed on. The AGPL extends that to people who use it
  over a network. Tell the person before installing, and do not install a
  plugin as a side effect of something else.
- **Ultralytics may send usage statistics** unless that is turned off
  (`yolo settings sync=False`). The engine sends nothing.
- **The engine does not need either.** Its own effects run through `ffmpeg` and
  its own detectors are Apache-2.0. Measure before claiming a plugin is better
  for the job at hand.

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
[docs/ADAPTERS.md](https://github.com/constructive-io/constructive-media/blob/main/docs/ADAPTERS.md).
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
| `constructive-media` | [constructive-media](https://github.com/constructive-io/constructive-media) | Setting up and using the engine |
| `conv` | [conversation-format](https://github.com/constructive-io/conversation-format) | Writing and checking the `.conv` scripts the engine renders |
