# constructive-media-plugins

<p align="center" width="100%">
  <img height="250" src="https://raw.githubusercontent.com/constructive-io/constructive/refs/heads/main/assets/outline-logo.svg" alt="Constructive" />
</p>

<p align="center" width="100%">
  Adapters for <a href="https://github.com/constructive-io/constructive-media">Constructive Media</a> whose tools carry a copyleft license.
</p>

Constructive Media is MIT licensed, and everything it contains or needs is MIT,
Apache-2.0 or BSD. Some tools worth using are not. An adapter for one of them
lives here, under that tool's license, as a package of its own that you choose
to install. The engine never imports any of them by name: it finds an installed
plugin through an entry point, and works the same without one.

| Plugin | Fills the slot | Chosen as | Tool | License |
| --- | --- | --- | --- | --- |
| [`pedalboard-effects`](./plugins/pedalboard-effects) | `effects` | `pedalboard` | [pedalboard](https://github.com/spotify/pedalboard) | GPL-3.0-only |
| [`ultralytics-detection`](./plugins/ultralytics-detection) | `detection` | `ultralytics` | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0-only |

## Install

Install the engine first, then the plugins you want, into the same environment:

```sh
pip install "./constructive-media[all]"
pip install ./constructive-media-plugins/plugins/pedalboard-effects
pip install ./constructive-media-plugins/plugins/ultralytics-detection

me adapters
```

`me adapters` lists every adapter the engine can see, with its license and
where it came from:

```
effects: audio effects
 * ffmpeg       MIT           builtin                        Reverb, delay, compression and gain through the ffmpeg program
   pedalboard   GPL-3.0-only  constructive-media-pedalboard  Reverb, delay, compression and gain with pedalboard's studio effects
```

## Use

A plugin is never the default. Installing one changes nothing until you choose
it, by slot:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_EFFECTS=pedalboard
export CONSTRUCTIVE_MEDIA_ADAPTER_DETECTION=ultralytics
```

## Licenses

There is no license for the repository as a whole. Each plugin is a separate
package under the license in its own folder, the same one as the tool it
drives.

Installing a plugin does not change the engine's license. It does change what
you are running: a program that includes a GPL or AGPL part is covered by those
terms when you pass it on, and the AGPL extends that to people who use it over
a network. Read the license of any plugin you install before you ship what you
build with it. This is a description, not legal advice.

No model weights, voices, audio or footage are committed here. A model a plugin
downloads comes from whoever publishes it, under their license.

## Writing a plugin

A plugin is an ordinary Python package that declares an entry point in the
`constructive_media.adapters` group and returns its adapters, each with its
slot, its name and its license stated honestly. What an adapter must do is set
by its slot. Both are described in the engine's
[docs/ADAPTERS.md](https://github.com/constructive-io/constructive-media/blob/main/docs/ADAPTERS.md).

## Tests

Each plugin has its own, in standard-library `unittest`. With the engine and
the plugin installed:

```sh
cd plugins/pedalboard-effects && python -m unittest discover -s tests -t .
```

Sound and images are made by the tests. No trained model is downloaded.

## Related

- [constructive-media](https://github.com/constructive-io/constructive-media): the engine.
- [conv-spec](https://github.com/constructive-io/conv-spec): the `.conv` script format the engine renders.
- [Constructive](https://github.com/constructive-io/constructive): the secure-by-default Postgres platform this is part of.
