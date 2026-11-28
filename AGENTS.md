# media-engine-plugins — agent guide

Adapters for [Constructive Media](https://github.com/constructive-io/media-engine)
whose tools, or the models they run, are not MIT, Apache-2.0 or BSD licensed. See
[README.md](./README.md).

How plugins are installed, chosen and written is a skill in this repo:
`.agents/skills/media-engine-plugins/SKILL.md`. Read it before installing a plugin for
somebody or writing one. Using the engine itself is the `constructive-media` skill in the
engine's repository, and writing `.conv` scripts is the `conv` skill in
[conversation-format](https://github.com/constructive-io/conversation-format).

## Rules that are specific to this repository

- **One plugin, one folder, one license.** Each folder under `plugins/` is a package of
  its own with a `LICENSE`, a `pyproject.toml` and tests. Its license is the license of
  the tool it drives, stated as an SPDX identifier in `pyproject.toml` and on the
  `Adapter` it returns. Plugins do not import one another.
- **A model under terms of its own is named as that.** Where the libraries a plugin
  uses are permissive and it is the model that is not (`ltx-video`), the plugin's code
  is MIT, the `Adapter` states the model's terms as a `LicenseRef-` identifier, and the
  plugin's README sets those terms out version by version, from the published texts,
  as a description and not as advice. Read the text before writing about it: the terms
  of one model have differed from one version to the next.
- **Nothing here goes into the engine.** Most code written here is copyleft, and an
  adapter that is MIT still runs something the engine may not hold. Do not copy either
  into `constructive-media`, and do not copy the engine's interfaces here: import them.
- **A plugin is never the default** for its slot, and installing one changes nothing
  until it is chosen with `CONSTRUCTIVE_MEDIA_ADAPTER_<SLOT>`.
- **A tool's library is imported only where it is used**, so that listing adapters
  costs nothing.
- **No content.** No model weights, voices, audio, images or footage, and no binaries.
  Tests make their own sound and pictures and never download a trained model.
- **Nothing personal or local.** No real people, project names, machine paths or
  credentials in code, comments, tests, docs or commit messages.
- Tests are standard-library `unittest`, run from the plugin's folder with the engine and
  the plugin installed: `python -m unittest discover -s tests -t .`
