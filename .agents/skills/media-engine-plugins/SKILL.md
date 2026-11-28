---
name: media-engine-plugins
description: Plugins for Constructive Media — optional adapters, each a package of its own that fills one of the engine's slots. Use when asked to "install a media engine plugin", "use pedalboard effects", "use Ultralytics YOLO for detection", "use Seed-VC", "convert a voice", "say a recording again in another voice", "install voice conversion", "make a still move", "animate a picture", "image to video", "use LTX-Video", "take a voice out of a song", "separate a voice from music", "remove the music from under a voice", "use Demucs", "lip sync a video", "make a face say a recording", "make a picture talk", "use LatentSync", "switch the effects backend", "switch the detector", "write a plugin for the media engine", "add an adapter", or when a tool or a model somebody wants is not MIT, Apache-2.0 or BSD licensed.
compatibility: Python 3.11+, with constructive-media installed in the same environment
metadata:
  author: constructive-io
  version: "0.5.0"
---

# Media engine plugins

[Constructive Media](https://github.com/constructive-io/media-engine) is
MIT licensed and contains only MIT, Apache-2.0 and BSD code. A tool under
another license, or a model published under terms of its own, is driven by a
plugin from this repository: a separate package that says what it is under,
which the engine finds once it is installed and uses only when it is chosen.

## When to apply

- Installing a plugin, choosing it, or checking which adapter a slot is using.
- Making a still move, which the engine's own video adapter does not do.
- Taking a voice out from over music, which nothing in the engine does.
- Making the face in a video say a recording, which nothing in the engine does.
- Writing an adapter for a tool or a model whose license the engine cannot hold.
- Answering what installing a plugin does to the license of what is built.

For using the engine itself (setting it up, a `.conv` script to sound, footage
to a cut), use the `constructive-media` skill in the engine's repository.

## The plugins

| Plugin | Fills the slot | Chosen as | Tool | License |
| --- | --- | --- | --- | --- |
| [`pedalboard-effects`](../../../plugins/pedalboard-effects) | `effects` | `pedalboard` | pedalboard | GPL-3.0-only |
| [`ultralytics-detection`](../../../plugins/ultralytics-detection) | `detection` | `ultralytics` | Ultralytics YOLO | AGPL-3.0-only |
| [`seedvc-conversion`](../../../plugins/seedvc-conversion) | `voice_conversion` | `seedvc` | Seed-VC | GPL-3.0-only |
| [`ltx-video`](../../../plugins/ltx-video) | `video` | `ltx` | LTX-Video | Code MIT; the model under Lightricks' terms |
| [`demucs-separation`](../../../plugins/demucs-separation) | `separation` | `demucs` | Demucs | Code MIT; the weights for research only |
| [`latentsync-lipsync`](../../../plugins/latentsync-lipsync) | `lip_sync` | `latentsync` | LatentSync | Code MIT; the weights under use restrictions, the face models for non-commercial research |

Each wraps the real library, which it installs as a dependency. Nothing is
reimplemented here. The one exception is `latentsync-lipsync`: LatentSync is
not a package, so the plugin installs none of it and runs a checkout of it
with that checkout's own Python.

## Install, choose, check

Into the environment the engine is installed in:

```sh
pip install ./plugins/pedalboard-effects
pip install ./plugins/ultralytics-detection
pip install ./plugins/seedvc-conversion      # Python 3.11 or 3.12 only
pip install ./plugins/ltx-video              # about 26 GB of weights when first used
pip install ./plugins/demucs-separation      # about 84 MB of weights when first used
pip install ./plugins/latentsync-lipsync     # and a checkout of LatentSync; about 6 GB of weights when first used

media adapters   # every slot, what can fill it, its license and where it came from
```

Installing changes nothing. A plugin is never the default; choose it by slot:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_EFFECTS=pedalboard
export CONSTRUCTIVE_MEDIA_ADAPTER_DETECTION=ultralytics
export CONSTRUCTIVE_MEDIA_MODEL_DETECTION=yolo11n    # a name Ultralytics publishes, or a model file
export CONSTRUCTIVE_MEDIA_ADAPTER_VOICE_CONVERSION=seedvc
export CONSTRUCTIVE_MEDIA_ADAPTER_VIDEO=ltx          # or --adapter ltx, for one piece
export CONSTRUCTIVE_MEDIA_ADAPTER_SEPARATION=demucs  # or --adapter demucs, for one recording
export CONSTRUCTIVE_MEDIA_ADAPTER_LIP_SYNC=latentsync # or --adapter latentsync, for one video
```

`media adapters` marks the adapter in use with `*`. Unset the variable to go back
to the engine's own. The engine has none of its own for `voice_conversion`:
`media revoice` and `media talk --quick` work only with a plugin chosen. It has
none for `separation` either: `media separate` works only with one. Nor for
`lip_sync`: `media lipsync` works only with one.

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

## A still made to move

```sh
media generate video "fog drifting over the water, the boat rocking gently" --from harbour.png --adapter ltx
media generate video "a paper lantern swinging under a porch in light rain" --seconds 4 --adapter ltx
```

- **Say what the model is under before installing it.** The code is MIT. The
  model is Lightricks', under terms that differ by version: 0.9 and 0.9.1 are
  for research alone and this adapter does not run them; the default, 0.9.5,
  forbids a list of uses and asks that what it makes be said to be machine
  made; 0.9.6 and later point entities with at least $10,000,000 of annual
  revenue to a paid license. The plugin's README has the table and the links.
  Describe the terms; do not rule on whether somebody's use is allowed.
- **It fetches about 26 GB the first time, and needs about 29 GB of memory.**
  Check the disk and say so before the first run. Do not call it hung.
- **Two seconds take about a minute** on Apple silicon and four take about
  two, loading included. A piece is 24 frames a second, with no sound.
- **Describe movement, not the picture.** The picture is the first frame; the
  words say what happens after it. Expect gentle movement: fog, water, a slow
  push of the camera.
- **No `--size` with a picture** unless a size is wanted: the piece takes the
  picture's own shape. A size is filled and cropped, never squashed. Sides
  are multiples of 32.
- **A clean picture barely moves**, so the adapter roughens it first, as a
  video encoder would. If a piece from a picture holds still, raise `rough`
  (default 12, up to 31) in `CONSTRUCTIVE_MEDIA_EXTRA_VIDEO` before raising
  `steps`.
- **Its settings are read only when it is the adapter chosen** with
  `CONSTRUCTIVE_MEDIA_ADAPTER_VIDEO=ltx`. With `--adapter ltx` alone it runs
  on its defaults.
- **Somebody's picture is somebody's.** The terms forbid impersonating a
  person without their consent. Do not make a picture of a real person move
  without their yes.

## A voice taken out from over music

```sh
media separate song.wav parts --adapter demucs     # parts/voice.wav and parts/rest.wav
CONSTRUCTIVE_MEDIA_EXTRA_SEPARATION='{"parts": "all"}' media separate song.wav parts --adapter demucs
```

- **Say what the weights are under before installing it, and again before
  what it makes is used.** The code is MIT. No license was published for the
  weights: their author says they "are not covered by the MIT license, and
  are provided only for scientific purposes", and that he cannot grant one.
  The plugin's README has each statement and its link. Describe that; do not
  rule on whether somebody's use is allowed, and do not reach for this plugin
  for work that is plainly commercial without the person knowing.
- **It is small and quick.** About 84 MB of weights, fetched once. Three
  minutes of sound took 7 seconds on Apple silicon and 35 on its CPU.
- **Memory grows with the recording**: about 2 GB up to three minutes, about
  11 GB for half an hour. Cut a long recording first.
- **Some of the music stays.** In one trial it came out 13 dB further under
  the voice, not gone, and the voice lost a little of itself. Have the
  person listen to `voice.wav` before a voice is cloned from it.
- **Read what it prints.** A level is printed for each part. "Hardly anything
  came out as the voice" means no voice was found; "hardly anything came out
  but the voice" means the recording can be used as it is.
- **`parts: "all"`** writes `drums.wav`, `bass.wav` and `other.wav` beside
  the voice. Everything that is not a voice, drums or a bass is `other`.
- **All who speak or sing are one voice to it.** It does not tell people
  apart; `media diarize` does that, afterwards.
- **The recording is somebody's, and so is the voice in it.** Pulling a
  voice out of a song gives nobody the right to use either.

## A face made to say a recording

```sh
export CONSTRUCTIVE_MEDIA_EXTRA_LIP_SYNC='{"repo": "~/LatentSync"}'
media lipsync face.mp4 line.wav out/face-says-line.mp4 --adapter latentsync
```

- **Whose face is it?** Ask first. A face a model made is fine. A real
  person's face needs that person's yes to these words; without it, do not
  do it, whoever asks and whatever for. The weights' terms forbid false
  content made to harm somebody.
- **Say what it is under before setting it up.** The plugin's code is MIT
  and LatentSync's is Apache-2.0. The weights are tagged Open RAIL++-M, which
  lists uses they may not be put to. The InsightFace models that find the
  face are, in their makers' words, "for non-commercial research purposes
  only", and LatentSync does not run without them. The plugin's README has
  each in its makers' words. Describe the terms; do not rule on whether
  somebody's use is allowed, and do not reach for this plugin for work that
  is plainly commercial without the person knowing.
- **It installs nothing of LatentSync.** It needs a checkout
  (`git clone https://github.com/bytedance/LatentSync`) and a Python in it
  with what LatentSync needs, at `.venv` or named with `python`. The
  plugin's README has the list that installed on Apple silicon;
  LatentSync's own `requirements.txt` is for an NVIDIA GPU. Do not edit the
  checkout, and do not install LatentSync's requirements into the engine's
  environment.
- **It fetches about 6 GB the first time, and uses about 20 GB of memory.**
  Check the disk and say so before the first run. Do not call it hung.
- **About half a minute for each second of speech** on Apple silicon: five
  seconds of speech took 160 s. Sync a line at a time. Every frame is held at
  once, so a long clip runs out of memory before it runs out of patience.
- **Turn `deepcache` on once it has been looked at.** `"deepcache": true` in
  the options took the same five seconds in 84 s, and the frames could not be
  told apart from the slow run's. It needs the DeepCache package in
  LatentSync's Python, and says so by name when it is missing.
- **Start from a face that moves.** A still made into a video comes back
  stock still but for its mouth. Make the still move first with the `ltx`
  plugin, a little longer than the line, then sync it.
- **A face in every frame**, large and turned towards the camera. A frame
  with none stops it, and the command says so.
- **A video shorter than the speech is played forwards and then backwards**
  until the speech ends. The command prints a note when that happened: look
  at the join.
- **What comes back is 25 frames a second**, whatever the video was, and as
  long as the speech.
- **Look before saying it worked.** Pull a sheet of frames and look at the
  mouth: `ffmpeg -i OUT -vf fps=2,tile=6x2 -frames:v 1 sheet.png`.
- **It was tried on Apple silicon only.** On an NVIDIA GPU it makes the
  calls LatentSync's own script makes; say that it is untried there.

## Before recommending one

- **Say what the license means.** A program that includes a GPL part is covered
  by the GPL when it is passed on. The AGPL extends that to people who use it
  over a network. A model's terms can forbid uses, or commercial use, whoever
  runs it. Tell the person before installing, and do not install a plugin as a
  side effect of something else.
- **Ultralytics may send usage statistics** unless that is turned off
  (`yolo settings sync=False`). The engine sends nothing.
- **The engine does not need pedalboard or Ultralytics.** Its own effects run
  through `ffmpeg` and its own detectors are Apache-2.0. Measure before
  claiming a plugin is better for the job at hand. Voice conversion, video
  from a picture, a voice taken out from over music and a face made to say a
  recording are the things here the engine cannot do without a plugin.

## Writing a plugin

One folder under `plugins/`, one package, one license:

```
plugins/my-tool/
├── LICENSE              # the license of the tool it drives (MIT, where only the model has terms)
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
            license="GPL-3.0-only",     # the SPDX identifier, stated honestly; a model's own terms as LicenseRef-…
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
