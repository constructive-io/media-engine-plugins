# media-engine-ultralytics

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) detectors for
[Constructive Media](https://github.com/constructive-io/media-engine).

It fills the engine's `detection` slot under the name `ultralytics`: what is in
each frame, as labelled boxes.

```sh
pip install ./plugins/ultralytics-detection
export CONSTRUCTIVE_MEDIA_ADAPTER_DETECTION=ultralytics
export CONSTRUCTIVE_MEDIA_MODEL_DETECTION=yolo11n     # the default
```

The model is any detection model Ultralytics can load:

- a name it publishes (`yolo11n`, `yolo11s`, `yolo11m`, ...). Ultralytics
  downloads the file on first use, into `ultralytics/` in the engine's models
  folder;
- or the path of a model file of your own.

`CONSTRUCTIVE_MEDIA_DEVICE` chooses where it runs (`auto`, `cpu`, `cuda`,
`mps`). On one Apple silicon desktop `yolo11n` takes about 18 ms a frame on
the CPU and 10 ms on the GPU.

## Before you use it

- **The license reaches further than most.** Ultralytics' code and the models
  it publishes are AGPL-3.0. That covers a service other people use over a
  network, not only a program you hand them. Ultralytics sells a separate
  license for uses the AGPL does not fit. The engine's own detectors need
  none of this.
- **Ultralytics may send anonymous usage statistics and crash reports** unless
  you turn that off (`yolo settings sync=False`). The engine itself sends
  nothing anywhere.

## License

AGPL-3.0-only, the license Ultralytics carries. See [LICENSE](./LICENSE).
