# media-engine-latentsync

A face in a video made to say a recording, for
[Constructive Media](https://github.com/constructive-io/media-engine), through
[LatentSync](https://github.com/bytedance/LatentSync).

It fills the engine's `lip_sync` slot under the name `latentsync`. The engine
has the slot and the command, and nothing of its own to fill it with. This
does: a piece of video with a face in it and a recording of speech go in, and
the video comes back with the mouth saying the recording.

```sh
pip install ./plugins/latentsync-lipsync
export CONSTRUCTIVE_MEDIA_EXTRA_LIP_SYNC='{"repo": "~/LatentSync"}'

media lipsync face.mp4 line.wav out/face-says-line.mp4 --adapter latentsync
```

```
out/face-says-line.mp4: 5.2 s at 512x512, face.mp4 saying line.wav
  5.1 s of speech over 6.0 s of video, in 160.4 s by latentsync ByteDance/LatentSync-1.5
```

`--adapter latentsync` asks for it once. To have it do every lip sync:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_LIP_SYNC=latentsync
```

**The code in this folder is MIT. What it runs is under three other sets of
terms.** LatentSync's code is Apache-2.0. Its weights are published under the
Open RAIL++-M License, which lists what they may not be used for. And the
models its face finder fetches, from InsightFace, are in their makers' words
"for non-commercial research purposes only". That last is why this adapter is
a plugin and not part of the engine. Read [the terms](#the-terms) before you
use what it makes.

**The face is somebody's.** A model that moves a mouth will move anybody's.
Use a face a model made, or one whose owner said yes to these words, and say
that the result was made by a machine wherever it is shown.

## What it needs

LatentSync is not a package that can be installed. It is a repository that is
run from where it was cloned, and its script is written for an NVIDIA GPU. So
this adapter installs none of it. It is told where a checkout is, and runs
LatentSync there with that checkout's own Python, a run at a time, in a
folder made for the run. Nothing is written into the checkout and none of its
files are changed: what has to be different off an NVIDIA GPU is stood in for
as the model is loaded.

- **A checkout of LatentSync, and a Python that has what it needs.** On an
  NVIDIA GPU, LatentSync's own `requirements.txt` is the list. On Apple
  silicon that list does not install (one of its packages, `decord`, has no
  build for it), and this one did, into Python 3.11 with
  [uv](https://docs.astral.sh/uv/):

  ```sh
  git clone https://github.com/bytedance/LatentSync ~/LatentSync
  cd ~/LatentSync
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python torch torchvision "diffusers==0.32.2" "transformers==4.48.0" \
      accelerate einops omegaconf "opencv-python==4.9.0.80" python_speech_features librosa ffmpeg-python \
      imageio imageio-ffmpeg huggingface-hub "numpy==1.26.4" "kornia==0.8.0" soundfile \
      insightface onnxruntime matplotlib
  uv pip install --python .venv/bin/python --no-deps "DeepCache==0.1.1"    # for the `deepcache` option
  ```

  That is about 1.5 GB. The list is LatentSync's own, cut down to what
  moving a mouth uses, with `onnxruntime` in place of `onnxruntime-gpu`. It
  resolved to PyTorch 2.15 and InsightFace 2.1, where LatentSync pins 2.5.1
  and 0.7.3. A Python at `.venv` or `venv` in the checkout is found without
  being named; any other is named with `python`. The engine and this plugin
  do not go in it, and it need not be the Python the engine runs on.
- **It was tried on Apple silicon, on its graphics processor, and nowhere
  else**, with LatentSync as it stood at commit `a229c39`. On an NVIDIA GPU
  it makes the calls LatentSync's own script makes, and that has not been
  tried. It runs on the first of an NVIDIA GPU, Apple's graphics processor
  and the CPU that it finds, or on the one `CONSTRUCTIVE_MEDIA_DEVICE` names.
- **The CPU works, and is about nine times slower.** The same machine took
  609 s over 2.1 s of speech with `CONSTRUCTIVE_MEDIA_DEVICE=cpu`, where its
  graphics processor took 72 s, and used 10 GB of memory. The frames looked
  the same.
- **About 6 GB of weights, fetched the first time a mouth is moved.** The
  model is 5.1 GB, the speech model it listens through 76 MB and the picture
  model 335 MB, and they go where Hugging Face keeps models on your machine
  (`HF_HOME` or `HF_HUB_CACHE` if you set them). The face models are another
  600 MB, fetched by InsightFace itself, and go in the checkout's
  `checkpoints/auxiliary` when that folder is there and beside the engine's
  own models when it is not. Weights already in the checkout's `checkpoints`
  folder are not used unless `checkpoint` names them, except the speech
  model, which is.
- **About 20 GB of memory.** The runs below peaked between 17 and 20 GB,
  whatever the length. LatentSync's makers say 8 GB of video memory is the
  least the 1.5 model runs in; on Apple silicon it runs at full precision,
  which is where the rest goes. Every frame is held at once as well, so a
  long clip or a large picture adds to it: move a mouth a line at a time.
- **`ffmpeg`**, which the engine needs already, on the `PATH`: LatentSync
  calls it by name.

## How well, and how fast

One trial, on one Apple silicon desktop, with everything already on disk. The
face was one a model drew, 512 pixels square, made to move a little by the
[`ltx` plugin](../ltx-video) for six seconds; the speech was one of the
engine's built-in voices.

| Speech | Steps | `deepcache` | Took | Of that, loading |
| --- | --- | --- | --- | --- |
| 2.1 s | 20 | no | 72 s | 5 s |
| 5.1 s | 20 | no | 160 s | 5 s |
| 5.1 s | 30 | no | 227 s | 5 s |
| 5.1 s | 20 | yes | 84 s | 5 s |

So about half a minute for each second of speech at the default, and about a
quarter of a minute with `deepcache`, which is how LatentSync's own
`inference.sh` runs it. In the frames that were looked at, two a second from
each, the run with `deepcache` and the run without could not be told apart;
it used as much memory. It is off unless asked for because it needs one more
package, and this is one clip: look at yours. Half as many steps again took
40% longer, and that could not be told apart from 20 steps either. The first run of all, which fetched the speech
model and the face models on the way, took 168 s.

The mouth opens and closes with the words. The rest of the face is the
video's own, frame for frame. The mouth is a little softer than the face
around it: the model works on a face 256 pixels square, and it is laid back
into the frame at the size the face was. No coloured specks, of the kind
half precision paints on Apple silicon, came out in any run.

What comes back:

- **runs as long as the speech**, with the recording under it as it was
  handed in.
- **is 25 frames a second**, whatever the video was. LatentSync reads every
  video at that rate.
- **is the size the video was**, at the one size tried.
- **has a face in every frame, or is not made.** A frame with no face in it
  stops LatentSync, and the command says so.

A video that ends before the speech does is played forwards and then
backwards by LatentSync until the speech is over, and `media lipsync` says
when that happened:

```
out/still-says-line.mp4: 5.2 s at 512x512, still.mp4 saying line.wav
  5.1 s of speech over 3.0 s of video, in 167.7 s by latentsync ByteDance/LatentSync-1.5
  the video ran out 2.1 s before the speech did: what the picture does after that is the adapter's doing, so look at it
```

A face that moves a little to begin with comes out better than a still: a
still made into a video holds stock still but for its mouth.

## Options

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_LIP_SYNC=latentsync
export CONSTRUCTIVE_MEDIA_MODEL_LIP_SYNC=ByteDance/LatentSync-1.5
export CONSTRUCTIVE_MEDIA_EXTRA_LIP_SYNC='{"repo": "~/LatentSync", "deepcache": true, "steps": 20, "guidance": 1.5, "seed": 1247}'
```

| Option | What it is | Default |
| --- | --- | --- |
| `repo` | The folder LatentSync was cloned into | none: it must be named |
| `python` | The Python that has what LatentSync needs | `.venv` or `venv` in the checkout |
| `config` | The file in the checkout that says how large a face the model works on | `configs/unet/stage2.yaml` |
| `checkpoint` | A file of weights to run in place of the model's own | none |
| `steps` | How many times over each stretch of the picture is worked, from 1 to 100 | 20 |
| `guidance` | How hard the mouth is held to the speech, from 1 to 10. LatentSync's makers say more is better sync and may distort or jitter | 1.5 |
| `seed` | Where the model's chance starts: the same seed moves the same mouth the same way | 1247 |
| `deepcache` | `true` to work one step in three in full and have the two after it reuse most of that. About twice as fast here. Needs the [DeepCache](https://github.com/horseee/DeepCache) package (Apache-2.0) in LatentSync's Python, and is refused by name without it | `false` |

An option it does not know is refused, by name. The model and the options are
read when this is the adapter chosen for lip sync, or when none is chosen and
this one is asked for with `--adapter latentsync`.

## Which models

The default is `ByteDance/LatentSync-1.5`, with the configuration it was
published for, and it is the only one this was tried with. The model is the
name of a repository on Hugging Face that holds a `latentsync_unet.pt`.

`ByteDance/LatentSync-1.6` was trained on faces 512 pixels square, and is run
with `"config": "configs/unet/stage2_512.yaml"`. Its makers say it needs
18 GB of video memory where 1.5 needs 8. It was not tried here.

## The terms

This is a description of what has been published, read in November 2026. It
is not legal advice.

- **LatentSync's code** is under the Apache License 2.0, in the
  [LICENSE](https://github.com/bytedance/LatentSync/blob/main/LICENSE) of its
  repository.
- **LatentSync's weights.** The repository they are published in,
  [`ByteDance/LatentSync-1.5`](https://huggingface.co/ByteDance/LatentSync-1.5),
  holds no license text. Its card is tagged `license: openrail++` and says no
  more, and so is the card of 1.6.
  [Hugging Face's list of licenses](https://huggingface.co/docs/hub/repositories-licenses)
  gives that tag as the Open Rail++-M License, and points to
  [the text Stable Diffusion XL was published under](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md),
  the "CreativeML Open RAIL++-M License dated July 26, 2023". That text:
  - grants "a perpetual, worldwide, non-exclusive, no-charge, royalty-free,
    irrevocable copyright license to reproduce, prepare, publicly display,
    publicly perform, sublicense, and distribute the Complementary Material,
    the Model, and Derivatives of the Model";
  - says "You cannot use the Model and the Derivatives of the Model for the
    specified restricted uses", and "You shall require all of Your users who
    use the Model or a Derivative of the Model to comply with the terms of
    this paragraph (paragraph 5)";
  - says of what is made, "Licensor claims no rights in the Output You
    generate using the Model", and "You are accountable for the Output you
    generate and its subsequent uses";
  - and lists the restricted uses in its Attachment A, these among them: "In
    any way that violates any applicable national, federal, state, local or
    international law or regulation"; "For the purpose of exploiting, harming
    or attempting to exploit or harm minors in any way"; "To generate or
    disseminate verifiably false information and/or content with the purpose
    of harming others"; "To generate or disseminate personal identifiable
    information that can be used to harm an individual"; "To defame,
    disparage or otherwise harass others".

  It also says the "Licensor reserves the right to restrict (remotely or
  otherwise) usage of the Model in violation of this License". Nothing in it
  forbids commercial use. Read the whole of it: this is a few of its lines.
- **The picture model**,
  [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse),
  is tagged MIT on its card.
- **The speech model** is Whisper's, of which
  [its makers say](https://github.com/openai/whisper#license): "Whisper's
  code and model weights are released under the MIT License".
- **The face models.** LatentSync finds the face in each frame with
  InsightFace's `buffalo_l` models, which InsightFace's library fetches the
  first time it is used.
  [InsightFace's README](https://github.com/deepinsight/insightface#license)
  says: "The code of InsightFace is released under the MIT License. There is
  no limitation for both academic and commercial usage." And then: "The
  training data containing the annotation (and the models trained with these
  data) are available for non-commercial research purposes only." The
  [README of its Python library](https://github.com/deepinsight/insightface/tree/master/python-package)
  says the same of what it fetches: "The pretrained models provided with this
  library are for non-commercial research only".

In short:

- **Nobody has granted a license to use the face models for anything but
  non-commercial research, and LatentSync as it is published does not run
  without them.** If what you are making is for anything else, that is a
  question for InsightFace or for a lawyer, and this plugin is not the tool
  for work that cannot wait for the answer.
- **The weights' terms follow the weights, and keeping to them is on whoever
  runs them.** False content made to harm somebody, and defaming somebody,
  are both on the list of what they may not be used for.
- **The video and the recording are separate matters.** Moving a mouth does
  not make the face or the voice yours to use.

`media adapters` shows this adapter as
`LicenseRef-LatentSync-OpenRAIL-weights-and-research-only-face-models`, which
is not the name of a license: it is there so that nobody installs it thinking
that everything it runs is MIT. This repository holds no weights.

## Licenses

The adapter's code is [MIT](./LICENSE), and it installs nothing but the
engine. What goes into the Python that runs LatentSync is LatentSync's to
list and yours to install, each package under its own license. The models
are as above.

## Tests

With the engine and this plugin installed:

```sh
python -m unittest discover -s tests -t .
```

No trained model is run and none is fetched, and there is no checkout of
LatentSync. The adapter is tried against a script that stands where the
runner would be and answers as it would, and the runner is tried against
stand-ins for PyTorch and for LatentSync that say which device they were
built on. The videos are plain colours and the recordings are tones.
