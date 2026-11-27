# media-engine-pedalboard

Studio effects for [Constructive Media](https://github.com/constructive-io/media-engine),
through Spotify's [pedalboard](https://github.com/spotify/pedalboard).

It fills the engine's `effects` slot under the name `pedalboard`: the reverb,
delay, compression and gain a `.conv` script can ask for on a line, a
character or a track.

```sh
pip install ./plugins/pedalboard-effects
export CONSTRUCTIVE_MEDIA_ADAPTER_EFFECTS=pedalboard
```

## What it does differently

The engine's own backend runs the `ffmpeg` program, which has no algorithmic
reverb, so it builds a room from a few echoes. This one runs a real reverb: a
dense tail that dies away as a room does, wider on a stereo track than the
sound that went in.

| Reverb | Dies away in | As wet as, unless you say |
| --- | --- | --- |
| `room` | 0.9 s | 0.12 |
| `hall` | 1.6 s | 0.20 |
| `plate` | 1.4 s | 0.18 |
| `cathedral` | 4.5 s | 0.30 |

- `reverb-mix` and `delay-mix` are a balance between the sound as it came and
  the effect: 0 is all of one, 1 all of the other. With nothing wet, the sound
  comes out exactly as loud as it went in.
- An echo or a room is left to ring out. The file written runs longer than the
  one read by as long as that takes, and the engine's mixer is told how long,
  so a track's last clip is given the room it needs.
- A WAV file of floats is answered with one, so sound that passes full scale on
  its way into a mix is not clipped. Whole numbers keep their depth.
- It is as fast as the `ffmpeg` backend: on one Apple silicon desktop, a minute
  of sound through a hall takes about 75 ms, against about 95 ms.

## License

GPL-3.0-only, the license pedalboard carries. See [LICENSE](./LICENSE).
