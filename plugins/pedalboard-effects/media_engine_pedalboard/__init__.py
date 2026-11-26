"""Studio effects for Constructive Media, through pedalboard.

Fills the engine's `effects` slot under the name `pedalboard`: an algorithmic
reverb, a feedback delay, a compressor and a gain, run in this process.

pedalboard is licensed GPL-3.0, so this adapter is too, and that is why it
lives here and not in the engine.

    Copyright (C) 2026 Dan Lynch <pyramation@gmail.com>
    Copyright (C) 2026 Constructive <developers@constructive.io>

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License, version 3, as published
    by the Free Software Foundation. It is distributed WITHOUT ANY WARRANTY;
    see the LICENSE file beside this package.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constructive_media.adapters import Adapter, lazy
from constructive_media.effects import EffectError, Effects

__version__ = "0.1.0"

LICENSE = "GPL-3.0-only"

DEFAULT_DELAY_FEEDBACK = 0.3
DEFAULT_DELAY_MIX = 0.3

#: An effect is left to ring until it is this far below where it began...
RUNG_OUT_DB = -60.0
#: ...and never for longer than this.
LONGEST_RING = 12.0
#: Trailing sound quieter than this, against full scale, is not kept.
SILENCE = 10 ** (-80 / 20)

#: The reverb's own scale: a dry level of a half passes the sound as it came.
_UNITY_DRY = 0.5


@dataclass(frozen=True)
class Space:
    """A reverb preset.

    `size` is how big the room is and `damping` how fast its highs die, both
    from 0 to 1. `mix` is how wet the preset is when no mix is asked for, and
    `rings` how many seconds it takes to die away.
    """

    size: float
    damping: float
    width: float
    mix: float
    rings: float


SPACES: dict[str, Space] = {
    "room": Space(size=0.25, damping=0.5, width=0.8, mix=0.12, rings=0.9),
    "hall": Space(size=0.60, damping=0.5, width=1.0, mix=0.20, rings=1.6),
    "cathedral": Space(size=0.90, damping=0.4, width=1.0, mix=0.30, rings=4.5),
    "plate": Space(size=0.40, damping=0.15, width=0.9, mix=0.18, rings=1.4),
}


def adapters() -> list[Adapter]:
    """What this plugin gives the engine."""
    return [
        Adapter(
            slot="effects",
            name="pedalboard",
            factory=lazy("media_engine_pedalboard:PedalboardEffects"),
            license=LICENSE,
            summary="Reverb, delay, compression and gain with pedalboard's studio effects",
        )
    ]


class PedalboardEffects:
    """Reverb, delay, compression and gain, as one chain of pedalboard effects."""

    name = "pedalboard"

    def chain(self, effects: Effects) -> list[Any]:
        """The pedalboard effects for a request, in the order they run."""
        import pedalboard

        chain: list[Any] = []
        if effects.reverb is not None:
            space = _space(effects.reverb)
            wet = space.mix if effects.reverb_mix is None else effects.reverb_mix
            chain.append(pedalboard.Reverb(
                room_size=space.size, damping=space.damping, width=space.width,
                wet_level=wet, dry_level=_UNITY_DRY * (1 - wet),
            ))
        if effects.delay is not None:
            chain.append(pedalboard.Delay(
                delay_seconds=effects.delay,
                feedback=DEFAULT_DELAY_FEEDBACK if effects.delay_feedback is None else effects.delay_feedback,
                mix=DEFAULT_DELAY_MIX if effects.delay_mix is None else effects.delay_mix,
            ))
        if effects.compress:
            chain.append(pedalboard.Compressor(threshold_db=-18.0, ratio=3.0))
        if effects.gain is not None:
            chain.append(pedalboard.Gain(gain_db=effects.gain))
        return chain

    def ring(self, effects: Effects) -> float:
        """How many seconds a request goes on sounding after its source ends."""
        return ring(effects)

    def apply(self, source: Path, target: Path, effects: Effects) -> Path:
        import numpy as np
        import pedalboard
        from pedalboard.io import AudioFile

        source, target = Path(source), Path(target)
        chain = self.chain(effects)
        if not chain:
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            return target
        try:
            with AudioFile(str(source)) as audio:
                rate, kind = audio.samplerate, str(audio.file_dtype)
                dry = audio.read(audio.frames)
        except Exception as error:  # noqa: BLE001 - whatever the reader raises, say which file
            raise RuntimeError(f"pedalboard could not read {source.name}: {error}") from error

        # Room to ring: an echo or a room goes on after the sound that set it off.
        tail = np.zeros((dry.shape[0], int(ring(effects) * rate)), dtype=np.float32)
        wet = pedalboard.Pedalboard(chain)(np.concatenate([dry, tail], axis=1), rate)
        wet = wet[:, :max(dry.shape[1], _sounding(wet))]

        # Never read and write one file at once.
        scratch = target.with_name(f".{target.stem}.fx{target.suffix}")
        try:
            how = {"bit_depth": _depth(kind)} if target.suffix.lower() in (".wav", ".flac", ".aiff") else {}
            with AudioFile(str(scratch), "w", rate, wet.shape[0], **how) as out:
                out.write(wet)
            os.replace(scratch, target)
        finally:
            scratch.unlink(missing_ok=True)
        return target


def ring(effects: Effects) -> float:
    """How many seconds a request goes on sounding after its source ends."""
    seconds = 0.0
    if effects.reverb is not None:
        seconds += _space(effects.reverb).rings
    if effects.delay is not None:
        feedback = DEFAULT_DELAY_FEEDBACK if effects.delay_feedback is None else effects.delay_feedback
        mix = DEFAULT_DELAY_MIX if effects.delay_mix is None else effects.delay_mix
        if mix > 0:
            # The first repeat, then one more for as long as the feedback keeps them above the floor.
            repeats = 1 + (math.ceil(RUNG_OUT_DB / (20 * math.log10(feedback))) if feedback > 0 else 0)
            seconds += effects.delay * repeats
    return min(seconds, LONGEST_RING)


def _space(preset: str) -> Space:
    try:
        return SPACES[preset]
    except KeyError:
        raise EffectError(
            f"unknown reverb {preset!r}; this backend offers: {', '.join(SPACES)}"
        ) from None


def _sounding(samples: Any) -> int:
    """How many samples in, the last sound loud enough to keep ends."""
    import numpy as np

    loud = np.nonzero(np.abs(samples).max(axis=0) > SILENCE)[0]
    return int(loud[-1]) + 1 if len(loud) else 0


def _depth(kind: str) -> int:
    """The sample format that loses nothing the source had.

    Floats stay floats, so sound that passes full scale is not clipped; deep
    whole numbers stay deep; anything else is written as 16 bit.
    """
    if kind.startswith("float"):
        return 32
    if kind in ("int24", "int32"):
        return 24
    return 16
