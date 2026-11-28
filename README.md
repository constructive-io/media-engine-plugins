# media-engine-plugins

<p align="center" width="100%">
  <img height="250" src="https://raw.githubusercontent.com/constructive-io/constructive/refs/heads/main/assets/outline-logo.svg" alt="Constructive" />
</p>

<p align="center" width="100%">
  Adapters for <a href="https://github.com/constructive-io/media-engine">Constructive Media</a> whose tools, or the models they run, are not MIT, Apache-2.0 or BSD licensed.
</p>

Constructive Media is MIT licensed, and everything it contains or needs is MIT,
Apache-2.0 or BSD. Some tools worth using are not: a library under the GPL, or
a model its makers publish under terms of their own. An adapter for one of them
lives here, as a package of its own that says what it is under and that you
choose to install. The engine never imports any of them by name: it finds an
installed plugin through an entry point, and works the same without one.

| Plugin | Fills the slot | Chosen as | Tool | License |
| --- | --- | --- | --- | --- |
| [`pedalboard-effects`](./plugins/pedalboard-effects) | `effects` | `pedalboard` | [pedalboard](https://github.com/spotify/pedalboard) | GPL-3.0-only |
| [`ultralytics-detection`](./plugins/ultralytics-detection) | `detection` | `ultralytics` | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0-only |
| [`seedvc-conversion`](./plugins/seedvc-conversion) | `voice_conversion` | `seedvc` | [Seed-VC](https://github.com/Plachtaa/seed-vc) | GPL-3.0-only |
| [`ltx-video`](./plugins/ltx-video) | `video` | `ltx` | [LTX-Video](https://github.com/Lightricks/LTX-Video) | Code MIT; the model under [Lightricks' terms](./plugins/ltx-video#the-models-terms) |
| [`demucs-separation`](./plugins/demucs-separation) | `separation` | `demucs` | [Demucs](https://github.com/adefossez/demucs) | Code MIT; the weights [for research only](./plugins/demucs-separation#the-weights-terms) |

## Install

Install the engine first, then the plugins you want, into the same environment:

```sh
pip install "./constructive-media[all]"
pip install ./media-engine-plugins/plugins/pedalboard-effects
pip install ./media-engine-plugins/plugins/ultralytics-detection
pip install ./media-engine-plugins/plugins/seedvc-conversion       # Python 3.11 or 3.12
pip install ./media-engine-plugins/plugins/ltx-video               # fetches about 26 GB of weights when first used
pip install ./media-engine-plugins/plugins/demucs-separation       # fetches about 84 MB of weights when first used

media adapters
```

`media adapters` lists every adapter the engine can see, with its license and
where it came from:

```
effects: audio effects
 * ffmpeg       MIT           builtin                  Reverb, delay, compression and gain through the ffmpeg program
   pedalboard   GPL-3.0-only  media-engine-pedalboard  Reverb, delay, compression and gain with pedalboard's studio effects
```

## Use

A plugin is never the default. Installing one changes nothing until you choose
it, by slot:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_EFFECTS=pedalboard
export CONSTRUCTIVE_MEDIA_ADAPTER_DETECTION=ultralytics
export CONSTRUCTIVE_MEDIA_ADAPTER_VOICE_CONVERSION=seedvc
export CONSTRUCTIVE_MEDIA_ADAPTER_VIDEO=ltx
export CONSTRUCTIVE_MEDIA_ADAPTER_SEPARATION=demucs
```

The first two take the place of an adapter the engine has of its own. The
third fills a slot the engine leaves empty: without it, `media revoice` and
`media talk --quick` say what to install. The fourth does something the
engine's own video adapter does not, which is to start from a picture, and it
can be asked for one piece at a time instead:

```sh
media generate video "fog drifting over the water" --from harbour.png --adapter ltx
```

The fifth fills another slot the engine leaves empty: without it,
`media separate` says what to install. With it, a voice is taken out from
over music:

```sh
media separate song.wav parts --adapter demucs     # parts/voice.wav, parts/rest.wav
```

## Licenses

There is no license for the repository as a whole. Each plugin is a separate
package under the license in its own folder. For a plugin that drives a
copyleft library, that is the library's license. For one that runs a model
published under terms of its own, the code is MIT and the model's terms are
named on the adapter and set out in the plugin's README.

Installing a plugin does not change the engine's license. It does change what
you are running: a program that includes a GPL or AGPL part is covered by those
terms when you pass it on, and the AGPL extends that to people who use it over
a network. A model's terms are about the model and what it is used for: some
forbid uses outright, some are for research alone, and some ask larger
companies to pay. Read the license of any plugin you install, and the terms of
any model it fetches, before you ship what you build with it. This is a
description, not legal advice.

No model weights, voices, audio or footage are committed here. A model a plugin
downloads comes from whoever publishes it, under their license.

## Writing a plugin

A plugin is an ordinary Python package that declares an entry point in the
`constructive_media.adapters` group and returns its adapters, each with its
slot, its name and its license stated honestly. What an adapter must do is set
by its slot. Both are described in the engine's
[docs/ADAPTERS.md](https://github.com/constructive-io/media-engine/blob/main/docs/ADAPTERS.md).

## Tests

Each plugin has its own, in standard-library `unittest`. With the engine and
the plugin installed:

```sh
cd plugins/pedalboard-effects && python -m unittest discover -s tests -t .
```

Sound and images are made by the tests. No trained model is downloaded.

## Working with an AI agent

The `media-engine-plugins` skill at `.agents/skills/media-engine-plugins/` covers
installing a plugin, choosing it, what its license means and writing a new one. Install it
where your agent looks with:

```sh
npx skills add https://github.com/constructive-io/media-engine-plugins --skill media-engine-plugins
```

The engine's repository has the `constructive-media` skill, for using the engine, and
conversation-format has the `conv` skill, for writing scripts.

## Related

- [media-engine](https://github.com/constructive-io/media-engine): the engine.
- [conversation-format](https://github.com/constructive-io/conversation-format): the `.conv` script format the engine renders.
- [Constructive](https://github.com/constructive-io/constructive): the secure-by-default Postgres platform this is part of.
