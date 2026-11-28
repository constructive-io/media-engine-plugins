# media-engine-ltx

Video from words, or from a still made to move, for
[Constructive Media](https://github.com/constructive-io/media-engine), through
[LTX-Video](https://github.com/Lightricks/LTX-Video).

It fills the engine's `video` slot under the name `ltx`. The engine's own
video adapter starts from words and nothing else. This one starts from words,
or from a picture and words: the picture is the first frame, and the words say
what happens next.

```sh
pip install ./plugins/ltx-video

media generate video "fog drifting over the water, the boat rocking gently" --from harbour.png --adapter ltx
media generate video "a paper lantern swinging under a porch in light rain" --seconds 4 --adapter ltx
```

`--adapter ltx` asks for it once. To have it make every piece of video:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_VIDEO=ltx
```

**The code in this folder is MIT. The model it runs is not.** Lightricks
publish LTX-Video under terms of their own, which differ by version, forbid
some uses outright and ask larger companies to buy a license for the later
versions. That is why this adapter is a plugin and not part of the engine.
Read [the model's terms](#the-models-terms) before you use what it makes.

## What it needs

- **PyTorch and diffusers 0.33 or later**, which this installs. It runs on
  Apple silicon, on an NVIDIA GPU or on the CPU, whichever it finds first in
  that order, or on the one `CONSTRUCTIVE_MEDIA_DEVICE` names. It was tried
  on Apple silicon and nowhere else.
- **About 26 GB of weights, fetched the first time a piece is made.** They go
  where Hugging Face keeps models on your machine (`HF_HOME` or `HF_HUB_CACHE`
  if you set them), and are fetched once. About 19 GB of that is the part
  that reads the words.
- **About 29 GB of memory** while a piece is made, measured on Apple silicon,
  for two seconds as for four.
- **`ffmpeg`**, which the engine needs already.

## How long it takes

On one Apple silicon desktop with the model already on disk. Each time is for
the whole command, of which loading the model is 11 to 17 seconds.

| Asked for | Made | Took |
| --- | --- | --- |
| Two seconds from words | 49 frames at 704x480 | 67 s |
| Two seconds from a picture of 1344x768 | 49 frames at 768x448 | 67 to 75 s |
| Four seconds from words | 97 frames at 704x480 | 110 s |

Every piece is 24 frames a second and has no sound.

## A still made to move

- **The picture is the first frame.** What the words describe happens after
  it, so describe movement: what drifts, who turns, where the camera goes.
- **The piece takes the picture's shape.** With no `--size`, it is made at
  about the area the model does well at (704x480) in the proportions of the
  picture: a picture of 1344x768 becomes video of 768x448. With `--size`, the
  picture is scaled until it covers that size and what hangs over is cut off
  evenly. It is never squashed.
- **The picture is roughened before the model sees it.** The model learned
  from video, and a picture cleaner than any frame of video it holds nearly
  still. Here the same picture, words and seed moved about three times as far
  between the first frame and the last once the picture had been through a
  video encoder. The adapter does that by default, with ffmpeg's own MPEG-4
  encoder, which every ffmpeg has. `rough` sets how hard, and 0 turns it off.
- **Movement is gentle.** In the trials here a still became fog thinning,
  water moving and a slow push of the camera. It did not become action.

## Sizes and lengths

- Each side is a multiple of 32. Anything else is refused with the nearest
  size that would do.
- A piece is 8n + 1 frames long, so `--seconds` is taken to the nearest such
  length: 2 seconds is 49 frames and 4 seconds is 97.
- Without `--size` or a picture, a piece is 704x480. Without `--seconds`, it
  runs 2 seconds.

## Options

`CONSTRUCTIVE_MEDIA_MODEL_VIDEO` names the model and
`CONSTRUCTIVE_MEDIA_EXTRA_VIDEO` is a JSON object of options. **Both are read
only when `ltx` is the adapter chosen for video** with
`CONSTRUCTIVE_MEDIA_ADAPTER_VIDEO`, so that the engine's own adapter can keep
its settings on the same machine. Asked for once with `--adapter ltx`, it runs
with these defaults:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_VIDEO=ltx
export CONSTRUCTIVE_MEDIA_MODEL_VIDEO=Lightricks/LTX-Video-0.9.5
export CONSTRUCTIVE_MEDIA_EXTRA_VIDEO='{"steps": 25, "fps": 24, "guidance": 3.0, "rough": 12}'
```

| Option | What it is | Default |
| --- | --- | --- |
| `steps` | How many passes the model makes. `--steps` goes over it for one piece. Lightricks' own examples use 40; here 40 took half as long again and moved a still no further | 25 |
| `fps` | Frames a second. The model was made for 24 | 24 |
| `guidance` | How closely the piece is held to the words | 3.0 |
| `rough` | How hard a starting picture is squeezed through a video encoder first, from 1 to 31. 0 leaves it as it is | 12 |
| `negative` | What the piece should not look like, in words | `worst quality, inconsistent motion, blurry, jittery, distorted` |

An option it does not know is refused, by name.

## Which models

The default is [`Lightricks/LTX-Video-0.9.5`](https://huggingface.co/Lightricks/LTX-Video-0.9.5),
and it is the only one this was tried with.

- **Before 0.9.5 is not run.** A model laid out as 0.9 or 0.9.1 were is
  refused once it has loaded, with a message that says so. Those two are also
  the versions licensed for research alone.
- **Later models were not tried here.** Lightricks publish them in the same
  layout, and they may load by name. They are far larger: the 0.9.7 distilled
  model is about 48 GB and the 0.9.8 one about 93 GB.

## The model's terms

This is a description of what Lightricks publish, not legal advice. The texts
are the authority, and they are short enough to read.

| Version | Published as | In short |
| --- | --- | --- |
| 0.9, 0.9.1 | [RAIL-M License, November 22, 2024](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.license.txt) | For "academic or research purposes only". This adapter does not run them |
| 0.9.5 | [Open RAIL-M License, March 5, 2025](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | The license to copy and pass on is no longer limited to a purpose. A list of uses is forbidden |
| 0.9.6 and later | [LTXV Open Weights License 0.X, April 15, 2025](https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt) | Use "for any purpose", subject to a list of forbidden uses. Entities with annual revenues of at least $10,000,000 are pointed to a paid commercial license |

What holds for the default, 0.9.5:

- **Some uses are forbidden whoever you are.** Among them: breaking a law,
  harming minors, spreading false information to harm others, defaming or
  harassing, impersonating somebody without their consent, and giving medical
  advice. The list is Attachment A of the license.
- **What it makes has to be said to be machine made.** The license forbids
  spreading what the model generates "without expressly and intelligibly
  disclaiming that the information and/or content is machine generated". The
  engine records every piece as generated, with its words, model and seed.
  Saying so where you publish it is yours to do.
- **Lightricks claim no rights in what you make** with the model, and hold
  you accountable for it and for what you put in.
- **Passing the model on carries the restrictions with it.** Anybody you
  give the weights to, or serve the model to, has to be held to the same
  list. This repository holds no weights.
- **One line of that text is unsettled.** Its list of forbidden uses still
  opens with "Outside of the Permitted Purpose", a term the 0.9.5 text no
  longer defines. In 0.9 it meant research alone. If that matters for what
  you are doing, ask Lightricks or use a later version, whose text does not
  have the line.

What the model is loaded with is under licenses of its own. The part that
reads the words is Google's T5, which is Apache-2.0, and comes in the same
download.

## Licenses

The adapter's code is [MIT](./LICENSE). PyTorch, diffusers, transformers,
SentencePiece, protobuf, NumPy and Pillow, which it installs, are BSD,
Apache-2.0 and MIT-CMU. The model is under Lightricks' terms, above, and
`media adapters` shows it as `LicenseRef-Lightricks-LTX-Video` so that nobody
installs it thinking otherwise.

## Tests

With the engine and this plugin installed:

```sh
python -m unittest discover -s tests -t .
```

No model is loaded and none is fetched. A stand-in answers with frames of one
colour, and the pictures a piece starts from are made by ffmpeg.
