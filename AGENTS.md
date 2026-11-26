# media-engine-plugins — agent guide

Adapters for [Constructive Media](https://github.com/constructive-io/constructive-media)
whose tools are not MIT, Apache-2.0 or BSD licensed. See [README.md](./README.md).

## Rules that are specific to this repository

- **One plugin, one folder, one license.** Each folder under `plugins/` is a package of
  its own with a `LICENSE`, a `pyproject.toml` and tests. Its license is the license of
  the tool it drives, stated as an SPDX identifier in `pyproject.toml` and on the
  `Adapter` it returns. Plugins do not import one another.
- **Nothing here goes into the engine.** Code written here is copyleft. Do not copy it
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
