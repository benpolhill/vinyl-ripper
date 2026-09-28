import numpy as np
import pytest

from vinylrip import split as sp
from vinylrip.config import DetectionConfig


def env_from_spec(spec: list[tuple[float, float]], block_s: float = 0.05) -> np.ndarray:
    """spec: list of (seconds, level_db)."""
    out = []
    for secs, db in spec:
        out += [db] * int(round(secs / block_s))
    return np.asarray(out, dtype=float)


def params(**kw) -> sp.DetectionParams:
    base = sp.DetectionParams(threshold_db=-45, min_gap_s=1.5, min_track_s=5, min_sound_s=0.3,
                              pop_max_s=1.0, pop_gap_s=0.4, pad_s=0.0)
    for k, v in kw.items():
        setattr(base, k, v)
    return base


def test_three_tracks_with_pops():
    env = env_from_spec([
        (1.0, -75), (0.1, -6), (2.0, -60),      # noise, needle drop, lead-in
        (20, -18), (3, -60), (12, -18), (3, -60), (16, -18),
        (4, -60), (0.1, -6), (0.5, -75),        # lead-out, needle lift
    ])
    segs = sp.detect_segments(env, 0.05, params())
    assert len(segs) == 3
    assert segs[0].start == pytest.approx(3.1, abs=0.1)
    assert segs[0].length == pytest.approx(20, abs=0.1)
    assert segs[1].length == pytest.approx(12, abs=0.1)
    assert segs[2].end == pytest.approx(3.1 + 20 + 3 + 12 + 3 + 16, abs=0.1)


def test_short_gap_within_track_is_not_a_split():
    env = env_from_spec([(2, -60), (10, -18), (1.0, -60), (10, -18), (2, -60)])
    segs = sp.detect_segments(env, 0.05, params(min_gap_s=1.5))
    assert len(segs) == 1
    segs = sp.detect_segments(env, 0.05, params(min_gap_s=0.8))
    assert len(segs) == 2


def test_lock_groove_ticks_ignored():
    spec = [(2, -60), (10, -18)]
    for _ in range(10):
        spec += [(1.75, -60), (0.05, -20)]
    env = env_from_spec(spec)
    segs = sp.detect_segments(env, 0.05, params())
    assert len(segs) == 1
    assert segs[0].end == pytest.approx(12, abs=0.1)


def test_pop_attached_to_music_is_trimmed():
    env = env_from_spec([(1, -60), (0.2, -6), (0.6, -60), (10, -18), (0.6, -60), (0.2, -6), (1, -60)])
    segs = sp.detect_segments(env, 0.05, params())
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(1.8, abs=0.1)
    assert segs[0].end == pytest.approx(11.8, abs=0.1)


def test_padding_does_not_overlap_neighbours():
    env = env_from_spec([(1, -60), (10, -18), (0.3, -60), (10, -18), (1, -60)])
    segs = sp.detect_segments(env, 0.05, params(min_gap_s=0.2, pad_s=1.0))
    assert len(segs) == 2
    assert segs[0].end <= segs[1].start


def test_auto_threshold_tracks_noise_floor():
    env = env_from_spec([(30, -55), (30, -18)])
    cfg = DetectionConfig()
    thr = sp.resolve_threshold(env, cfg)
    assert thr == pytest.approx(-39.5, abs=1.0)
    cfg.silence_threshold_db = -33
    assert sp.resolve_threshold(env, cfg) == -33


def test_editing_helpers():
    segs = [sp.Segment(0, 10), sp.Segment(12, 20)]
    assert sp.merge(segs, 0) == [sp.Segment(0, 20)]
    assert sp.split(segs, 1, 15) == [sp.Segment(0, 10), sp.Segment(12, 15), sp.Segment(15, 20)]
    assert sp.delete(segs, 0) == [sp.Segment(12, 20)]
    assert sp.add(segs, 10.5, 11.5)[1] == sp.Segment(10.5, 11.5)
    with pytest.raises(ValueError):
        sp.add(segs, 5, 15)
    with pytest.raises(ValueError):
        sp.split(segs, 0, 11)
    assert sp.set_bounds(segs, 0, None, 11)[0] == sp.Segment(0, 11)


def test_noise_floor_ignores_brief_pre_drop_silence_and_little_gap_time():
    # 20 min side: 4 s of near-digital silence before the needle drops (0.3%), ~1.6% surface noise
    # in gaps, the rest music with a wide dynamic range.
    rng = np.random.default_rng(0)
    music = rng.normal(-25, 6, int(1180 / 0.05))
    env = np.concatenate([np.full(80, -80.0), np.full(400, -56.0) + rng.normal(0, 0.5, 400), music])
    floor = sp.noise_floor_db(env)
    assert floor == pytest.approx(-56, abs=1.5)


def test_auto_threshold_falls_back_without_silence():
    env = np.full(2000, -39.0) + np.random.default_rng(1).normal(0, 1.5, 2000)
    assert sp.auto_threshold(env, 15.0) == sp.FALLBACK_THRESHOLD_DB


def test_side_matching_prefers_clearly_better_side():
    from vinylrip.metadata import Side, Track
    from vinylrip.rip import better_matching_side, side_match_score
    a = Side(1, "A", [Track("x", length_s=200), Track("y", length_s=300)])
    b = Side(1, "B", [Track("p", length_s=250), Track("q", length_s=250), Track("r", length_s=100)])
    c = Side(2, "C", [Track("m", length_s=201), Track("n", length_s=299)])
    segs = [sp.Segment(0, 251), sp.Segment(253, 503), sp.Segment(505, 605)]
    assert side_match_score(segs, b) < side_match_score(segs, a)
    hit = better_matching_side(segs, 0, [1, 2], [a, b, c])
    assert hit is not None and hit[0] == 1
    # A vs C differ by a few seconds only: no suggestion (margin too small)
    segs2 = [sp.Segment(0, 200), sp.Segment(202, 502)]
    assert better_matching_side(segs2, 0, [1, 2], [a, b, c]) is None
    # no lengths known: no opinion
    d = Side(1, "D", [Track("z")])
    assert side_match_score(segs, d) is None
    # fits nothing at all: stay quiet
    segs3 = [sp.Segment(0, 8), sp.Segment(10, 17), sp.Segment(19, 25)]
    assert better_matching_side(segs3, 0, [1, 2], [a, b, c]) is None
