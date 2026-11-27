# media-engine-seedvc

Voice conversion for [Constructive Media](https://github.com/constructive-io/media-engine),
through [Seed-VC](https://github.com/Plachtaa/seed-vc).

It fills the engine's `voice_conversion` slot under the name `seedvc`: what
one voice said is said again in the voice of a clip. The words, their timing
and their tune are the recording's. The voice is the clip's. Nothing is
trained and nothing is written down: a clip of a few seconds is enough, and
no more than its first 25 seconds are listened to.

```sh
pip install ./plugins/seedvc-conversion
export CONSTRUCTIVE_MEDIA_ADAPTER_VOICE_CONVERSION=seedvc

media revoice take.wav out/take-as-ada.wav --voice ada      # a persona whose voice was cloned from a clip
media revoice take.wav out/take-as-ada.wav --like ada.flac  # or any recording of the voice
media talk --voice ada --quick af_heart                     # a quick voice, said again in a cloned one
```

Nothing in the engine fills this slot, so there is no adapter of its own to go
back to: without this one chosen, those commands say what to install.

## What it needs

- **Python 3.11 or 3.12.** Seed-VC needs a NumPy older than 2, and there is
  none for Python 3.13.
- **PyTorch**, which Seed-VC installs. It runs on an NVIDIA GPU, on Apple
  silicon or on the CPU, whichever it finds first in that order.
- **About 2.8 GB of weights, fetched the first time something is said.** They
  go where Hugging Face keeps models on your machine (`HF_HOME` or
  `HF_HUB_CACHE` if you set them), and are fetched once.

| Fetched | From | License |
| --- | --- | --- |
| The converter | [`Plachta/Seed-VC`](https://huggingface.co/Plachta/Seed-VC) | GPL-3.0 |
| What it hears words with | [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | Apache-2.0 |
| What it hears a voice with | [`funasr/campplus`](https://huggingface.co/funasr/campplus) | Apache-2.0 |
| What makes the sound | [`FunAudioLLM/CosyVoice-300M`](https://huggingface.co/FunAudioLLM/CosyVoice-300M) (`hift.pt`) | Apache-2.0 |

## How well, and how fast

On one Apple silicon desktop, a recording of 15.5 seconds said again in two
other voices, each from one clip. `alike` is how like the clip the result is,
as the engine's speaker model hears it; the recording itself was 0.27 and 0.25
alike them, and a voice cloned from a clip comes out about 0.8.

| `steps` | One voice: time | alike | Another: time | alike |
| --- | --- | --- | --- | --- |
| 4 | 1.1 s | 0.76 | 1.7 s | 0.45 |
| 10 | 1.5 s | 0.79 | 2.8 s | 0.38 |
| 30 | 3.0 s | 0.77 | 6.6 s | 0.39 |

- **Some voices convert well and some do not.** The first is as near as a
  clone. The second leans towards the voice and is not it. `media revoice`
  prints the number each time so you know which you have.
- **More steps did not make it more like the voice.** They take longer, and
  that is all the speaker model could tell. The default is 10.
- **A longer clip takes longer**: the second voice's clip was 25 seconds long
  and the first's 15.
- A sentence of 2.5 seconds took 0.4 seconds at 4 steps and 0.6 at 10.
- Loading the models takes about 4 seconds, once in a program. `media revoice`
  is a program for each recording. `media talk` loads them before it prints
  its address.

In a conversation, with the engine's `kokoro` adapter saying the words first,
the first sentence of an answer took 0.7 to 1.0 seconds, where that adapter
alone took 0.16.

## Options

`CONSTRUCTIVE_MEDIA_EXTRA_VOICE_CONVERSION` is a JSON object. These are the
defaults:

```sh
export CONSTRUCTIVE_MEDIA_EXTRA_VOICE_CONVERSION='{"steps": 10, "guidance": 0.7, "length": 1.0, "fp16": true}'
```

| Option | What it is | From, to |
| --- | --- | --- |
| `steps` | how many passes the model makes over what was said | 1 to 200 |
| `guidance` | how hard it is pulled towards the clip | 0 to 5 |
| `length` | what is written is this many times as long as the recording | 0.25 to 4 |
| `fp16` | run in half precision | `true`, `false` |

It runs the one model, the small one made for talking. Seed-VC has others, for
singing and for keeping a recording's pitch, and this adapter does not offer
them yet.

## Before you use it

- **The license.** Seed-VC's code and its weights are GPL-3.0, and so is this
  adapter. A program that includes them is covered by the GPL when it is
  passed on.
- **Whose voice it is.** This puts one person's words in another person's
  voice. Use a voice that is yours or whose owner said yes to this use, and a
  recording whose speaker said the same.
- **It is one conversion at a time.** A second asked for meanwhile waits.

## What this does about the package

Seed-VC is written as a program, and three things it does on being imported
are not what a library should do: it points Hugging Face's cache at a folder
called `checkpoints` wherever the program was started, it switches every
warning off for the whole program, and it loads all of its models again for
every sentence. The adapter puts the first two back as they were and keeps the
models once they are loaded. It was written for Seed-VC 0.4 and says so when
it meets another.

## License

GPL-3.0-only, the license Seed-VC carries. See [LICENSE](./LICENSE).
