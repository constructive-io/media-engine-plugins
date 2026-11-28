# media-engine-demucs

A voice taken out from over music, for
[Constructive Media](https://github.com/constructive-io/media-engine), through
[Demucs](https://github.com/adefossez/demucs).

It fills the engine's `separation` slot under the name `demucs`. The engine
has the slot and the command, and nothing of its own to fill it with. This
does: a recording of somebody speaking or singing over music is written again
as the voice, and as everything that was under it.

```sh
pip install ./plugins/demucs-separation

media separate song.wav parts --adapter demucs
```

```
song.wav: -13.6 dB, pulled apart in 1.7 s by demucs htdemucs
  voice   -16.2 dB  13.2 s  parts/voice.wav
  rest    -17.6 dB  13.2 s  parts/rest.wav
```

`--adapter demucs` asks for it once. To have it do every separation:

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_SEPARATION=demucs
```

**The code in this folder is MIT, and so is the code of Demucs. The weights
are not.** Their authors say the weights "are not covered by the MIT license,
and are provided only for scientific purposes", because part of the music the
models learned from could be used for research and nothing else. That is why
this adapter is a plugin and not part of the engine. Read
[the weights' terms](#the-weights-terms) before you use what it makes.

## What it needs

- **Demucs 4.1 and PyTorch**, which this installs. It runs on Apple silicon,
  on an NVIDIA GPU or on the CPU, whichever it finds first in that order, or
  on the one `CONSTRUCTIVE_MEDIA_DEVICE` names. It was tried on Apple silicon,
  on its graphics processor and on its CPU, and nowhere else.
- **About 84 MB of weights, fetched the first time a recording is pulled
  apart.** They go where Hugging Face keeps models on your machine (`HF_HOME`
  or `HF_HUB_CACHE` if you set them), and are fetched once. After that it runs
  with no network, and `HF_HUB_OFFLINE=1` keeps it from asking.
- **Memory that grows with the recording**, which is held whole, and its parts
  with it: about 2 GB for anything up to three minutes, and about 11 GB for
  half an hour. Cut a long recording before it is pulled apart.
- **`ffmpeg`**, which the engine needs already. Anything ffmpeg reads can be
  pulled apart, the sound of a video among it.

## How well, and how fast

One trial, on one Apple silicon desktop, with the weights already on disk: a
spoken voice mixed over music with drums in it, the music about 1 dB under
the voice.

| | In the recording | In `voice.wav` |
| --- | --- | --- |
| The voice | -16.1 dB | -16.2 dB |
| What is not the voice | -17.3 dB | -30.7 dB |

So the music fell about 13 dB further under the voice. It did not go. What is
left is quiet and can be heard in the pauses, and the voice has lost a little
of itself along the way: the second row is everything in the file that the
voice alone did not have, which is the music that stayed and the voice that
changed. A louder or denser piece of music leaves more.

| Recording | On the graphics processor | On the CPU |
| --- | --- | --- |
| 13 seconds | 1.7 s | 4.2 s |
| 3 minutes | 7 s | 35 s |
| 30 minutes | 60 s | not tried |

Each time has loading the model in it, which is under a second. The two
devices gave the same answer, and so did the same recording pulled apart
twice.

A voice with nothing under it comes back whole, with a `rest.wav` of next to
nothing, and `media separate` says so:

```
  hardly anything came out but the voice: nothing was found under it, and the recording may as well be used as it is
```

## The parts

The files are WAV, 24 bits, at 44,100 samples a second, which is the rate the
models work at whatever the recording's was. A recording of two channels
comes back as two, and a recording of one as one.

- **`voice.wav`** is what the model calls `vocals`: whoever speaks or sings.
  Two people talking over music are one voice to it. It does not tell
  speakers apart.
- **`rest.wav`** is everything else, added up.

Asked for every part, it writes the voice and each thing the model keeps
apart, under the model's name for it:

```sh
CONSTRUCTIVE_MEDIA_EXTRA_SEPARATION='{"parts": "all"}' media separate song.wav parts --adapter demucs
```

```
song.wav: -13.6 dB, pulled apart in 1.7 s by demucs htdemucs
  voice   -16.2 dB  13.2 s  parts/voice.wav
  drums   -20.7 dB  13.2 s  parts/drums.wav
  bass    -68.1 dB  13.2 s  parts/bass.wav
  other   -20.6 dB  13.2 s  parts/other.wav
```

The model was made for songs, and that is how it hears everything: what is
not a voice, drums or a bass goes to `other`, a crowd or the weather as much
as a guitar.

## Options

```sh
export CONSTRUCTIVE_MEDIA_ADAPTER_SEPARATION=demucs
export CONSTRUCTIVE_MEDIA_MODEL_SEPARATION=htdemucs
export CONSTRUCTIVE_MEDIA_EXTRA_SEPARATION='{"parts": "two", "shifts": 0, "overlap": 0.25}'
```

| Option | What it is | Default |
| --- | --- | --- |
| `parts` | `two` for the voice and the rest, `all` for the voice and each of the model's other parts | `two` |
| `shifts` | How many times over the recording is pulled apart, each from a start a little way off, and the answers averaged, up to 10. Each one starts at random, so two runs differ. 0 does it once and the same way every time. Here 2 took a third as long again and changed what was left of the music by a tenth of a decibel | 0 |
| `overlap` | How much each stretch the model works on shares with the next, from 0 to less than 1 | 0.25 |

An option it does not know is refused, by name. The model and the options are
read when this is the adapter chosen for separation, or when none is chosen
and this one is asked for with `--adapter demucs`.

## Which models

The default is `htdemucs`, the one Demucs itself runs when it is not told
which, and it is the only one this was tried with. Any model Demucs loads by
name that keeps a part called `vocals` should run:

| Model | What its authors say of it |
| --- | --- |
| `htdemucs` | "first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model." |
| `htdemucs_ft` | Fine-tuned: "separation will take 4 times more time but might be a bit better" |
| `htdemucs_6s` | Six parts, with `piano` and `guitar` kept apart as well |
| `hdemucs_mmi`, `mdx`, `mdx_extra`, `mdx_q`, `mdx_extra_q` | Earlier models |

A model that keeps no voice apart is refused once it has loaded, with the
parts it does keep. A name Demucs does not know is refused in a line.

## The weights' terms

This is a description of what the authors have published, not legal advice.
There is no license text for the weights. What there is, in the order it was
written:

- **The author of Demucs, in May 2022**, asked what license the pre-trained
  models are under
  ([facebookresearch/demucs, issue 327](https://github.com/facebookresearch/demucs/issues/327)):
  "The model weights are not covered by the MIT license, and are provided
  only for scientific purposes."
- **The reason**, from a contributor in the same thread in June 2024: "The
  models are trained using MusDB dataset, which requires the result model can
  only be used for research purpose."
- **The README of Demucs today**, under
  [License](https://github.com/adefossez/demucs#license): "Demucs code is
  released under the MIT license as found in the LICENSE file. The license of
  the weights is not clear: MusDB contains some data that were only usable
  for research purpose. I sadly cannot help granting a license, both coming
  from Meta custom data, or from the original MusDB data. The 6 source
  variant of htdemucs was trained only on Meta internal dataset, which likely
  means CC-NC-BY license is applicable, as it was the case for all models
  released by FAIR. Do not write me to ask about it, I won't be able to
  clarify the situation further, either for MusDB or Meta internal dataset."
- **The music itself.** [MUSDB18](https://sigsep.github.io/datasets/musdb.html)
  says of its tracks that they "can only be used for academic purposes", and
  that 46 of them come from a collection under Creative Commons
  BY-NC-SA 4.0.

In short:

- **Nobody has granted a license to use these weights for anything but
  research,** and the person who made them says he cannot.
- **That reaches what they make only as far as you judge it does.** Nothing
  published says what may be done with a voice the weights pulled out of a
  recording. If the answer matters for what you are doing, it is a question
  for a lawyer, and this plugin is not the tool for work that cannot wait for
  one.
- **The recording is a separate matter.** Pulling a voice out of a song does
  not make the voice or the song yours to use.

`media adapters` shows this adapter as
`LicenseRef-Demucs-weights-research-only`, which is not the name of a license:
it is there so that nobody installs it thinking the weights are MIT.

The repository the weights were first published from,
`facebookresearch/demucs`, is archived. Demucs is kept up by its author at
[`adefossez/demucs`](https://github.com/adefossez/demucs), which is where the
package this installs comes from. This repository holds no weights.

## Licenses

The adapter's code is [MIT](./LICENSE), and so is Demucs. Of what Demucs
installs, einops, julius and PyYAML are MIT, PyTorch and NumPy are BSD, and
huggingface-hub, safetensors and sphn are Apache-2.0. Two are neither, and
are installed because Demucs asks for them: **lameenc**, which writes MP3
files and is LGPL-3.0-or-later, and **tqdm**, which draws progress bars and
is MPL-2.0 and MIT. This adapter uses neither. The weights are as above.

## Tests

With the engine and this plugin installed:

```sh
python -m unittest discover -s tests -t .
```

No trained model is run and none is fetched. A stand-in answers with fixed
shares of what it is handed, and the recordings are tones. One test runs the
real Demucs on the small untrained model its own tests use, which is built in
memory.
