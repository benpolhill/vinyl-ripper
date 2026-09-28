"""Silence-based track detection on a level envelope, plus pop trimming.

The envelope is a 1-D array of RMS levels in dBFS, one value per block of
`block_s` seconds. Working on the envelope (not the raw samples) keeps memory
tiny and makes re-detection with different parameters instant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import DetectionConfig


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class DetectionParams:
    threshold_db: float
    min_gap_s: float
    min_track_s: float
    min_sound_s: float
    pop_max_s: float
    pop_gap_s: float
    pad_s: float

    @classmethod
    def from_config(cls, cfg: DetectionConfig, threshold_db: float) -> "DetectionParams":
        return cls(
            threshold_db=threshold_db,
            min_gap_s=cfg.min_gap_seconds,
            min_track_s=cfg.min_track_seconds,
            min_sound_s=cfg.min_sound_seconds,
            pop_max_s=cfg.pop_max_seconds,
            pop_gap_s=cfg.pop_gap_seconds,
            pad_s=cfg.pad_seconds,
        )


def noise_floor_db(env: np.ndarray, min_fraction: float = 0.01) -> float:
    """Level of the quietest 'mode' in the recording that covers at least `min_fraction` of it.

    Between-track gaps, lead-in and lead-out grooves all sit at the surface-noise level, which
    shows up as the lowest bump in a level histogram. Using the lowest sufficiently populated
    mode (rather than a percentile) works whether the side has 1% or 30% silence.
    """
    if env.size == 0:
        return -90.0
    clipped = np.clip(env, -90.0, 0.0)
    counts, edges = np.histogram(clipped, bins=np.arange(-90.0, 1.0, 1.0))
    smooth = np.convolve(counts, np.ones(3), mode="same")  # blocks within a 3 dB window
    min_count = max(2.0, min_fraction * env.size)
    for i in range(smooth.size):
        left = smooth[i - 1] if i > 0 else -1.0
        right = smooth[i + 1] if i + 1 < smooth.size else -1.0
        if smooth[i] >= min_count and smooth[i] >= left and smooth[i] >= right:
            return float(edges[i] + 0.5)
    return float(np.percentile(env, 5))


FALLBACK_THRESHOLD_DB = -45.0


def auto_threshold(env: np.ndarray, margin_db: float, lo: float = -65.0, hi: float = -30.0) -> float:
    """Threshold derived from the recording's own noise floor.

    Falls back to a fixed value when the estimate would classify almost everything as silence,
    which happens when a recording contains no gaps at all (e.g. an excerpt of one track).
    """
    thr = float(min(hi, max(lo, noise_floor_db(env) + margin_db)))
    if env.size and np.mean(env > thr) < 0.2:
        return FALLBACK_THRESHOLD_DB
    return thr


def resolve_threshold(env: np.ndarray, cfg: DetectionConfig) -> float:
    if cfg.silence_threshold_db == "auto":
        return auto_threshold(env, cfg.auto_margin_db)
    return float(cfg.silence_threshold_db)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as half-open [start, end) block indexes."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def detect_segments(env: np.ndarray, block_s: float, p: DetectionParams) -> list[Segment]:
    total_blocks = int(env.size)
    total_s = total_blocks * block_s
    if total_blocks == 0:
        return []

    def blocks(seconds: float) -> int:
        return max(1, int(round(seconds / block_s)))

    min_gap = blocks(p.min_gap_s)
    min_track = blocks(p.min_track_s)
    min_sound = blocks(p.min_sound_s)
    pop_max = blocks(p.pop_max_s)
    pop_gap = blocks(p.pop_gap_s)

    # 1. Raw bursts of sound; ignore bursts too short to be anything but a click.
    runs = [(s, e) for s, e in _runs(env > p.threshold_db) if (e - s) >= min_sound]

    # 2. Merge bursts separated by short gaps into regions (lists of bursts).
    regions: list[list[tuple[int, int]]] = []
    for run in runs:
        if regions and run[0] - regions[-1][-1][1] < min_gap:
            regions[-1].append(run)
        else:
            regions.append([run])

    # 3. Trim leading/trailing pops: short bursts isolated from the body by a small gap.
    trimmed: list[list[tuple[int, int]]] = []
    for region in regions:
        while len(region) >= 2 and (region[0][1] - region[0][0]) < pop_max and (region[1][0] - region[0][1]) >= pop_gap:
            region = region[1:]
        while len(region) >= 2 and (region[-1][1] - region[-1][0]) < pop_max and (region[-1][0] - region[-2][1]) >= pop_gap:
            region = region[:-1]
        trimmed.append(region)

    # 4. Drop regions too short to be a track.
    spans = [(r[0][0], r[-1][1]) for r in trimmed if (r[-1][1] - r[0][0]) >= min_track]

    # 5. Pad and convert to seconds, never overlapping neighbours.
    segments: list[Segment] = []
    for i, (s, e) in enumerate(spans):
        start = max(0.0, s * block_s - p.pad_s)
        end = min(total_s, e * block_s + p.pad_s)
        if i > 0:
            prev_end = spans[i - 1][1] * block_s
            start = max(start, prev_end)  # never eat into the previous track
        if i + 1 < len(spans):
            next_start = spans[i + 1][0] * block_s
            end = min(end, next_start)
        if segments and start < segments[-1].end:
            mid = (segments[-1].end + start) / 2
            segments[-1] = replace(segments[-1], end=mid)
            start = mid
        segments.append(Segment(start, end))
    return segments


# ----------------------------------------------------------------------- editing helpers

def merge(segments: list[Segment], index: int) -> list[Segment]:
    """Merge segment `index` with the following one (0-based)."""
    if not 0 <= index < len(segments) - 1:
        raise IndexError("nothing to merge with")
    out = list(segments)
    out[index : index + 2] = [Segment(out[index].start, out[index + 1].end)]
    return out


def split(segments: list[Segment], index: int, at: float) -> list[Segment]:
    seg = segments[index]
    if not seg.start < at < seg.end:
        raise ValueError(f"split point {at:.2f}s is outside segment ({seg.start:.2f}-{seg.end:.2f}s)")
    out = list(segments)
    out[index : index + 1] = [Segment(seg.start, at), Segment(at, seg.end)]
    return out


def delete(segments: list[Segment], index: int) -> list[Segment]:
    out = list(segments)
    del out[index]
    return out


def add(segments: list[Segment], start: float, end: float) -> list[Segment]:
    if end <= start:
        raise ValueError("end must be after start")
    out = sorted(segments + [Segment(start, end)], key=lambda s: s.start)
    for a, b in zip(out, out[1:]):
        if b.start < a.end:
            raise ValueError("segments would overlap")
    return out


def set_bounds(segments: list[Segment], index: int, start: float | None, end: float | None) -> list[Segment]:
    seg = segments[index]
    new = Segment(seg.start if start is None else start, seg.end if end is None else end)
    if new.end <= new.start:
        raise ValueError("end must be after start")
    out = list(segments)
    out[index] = new
    for a, b in zip(out, out[1:]):
        if b.start < a.end:
            raise ValueError("segments would overlap")
    return out
