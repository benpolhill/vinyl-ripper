"""Runs the whole file-based pipeline on a synthetic side. Needs ffmpeg on PATH."""

import shutil
import sys
import subprocess
from pathlib import Path

import pytest
from mutagen.flac import FLAC

from tests.make_side import build, write_wav
from vinylrip.capture import analyze_file
from vinylrip.config import Config
from vinylrip.encode import TrackJob, encode_track
from vinylrip import split as sp

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_detect_and_encode(tmp_path: Path):
    sig, expected = build([20, 12, 16])
    side = tmp_path / "side-A.wav"
    write_wav(side, sig)

    cfg = Config()
    audio = analyze_file(side, cfg)
    thr = sp.resolve_threshold(audio.envelope, cfg.detection)
    segs = sp.detect_segments(audio.envelope, audio.block_s, sp.DetectionParams.from_config(cfg.detection, thr))
    assert len(segs) == 3
    for seg, (s, e) in zip(segs, expected):
        assert seg.start == pytest.approx(s, abs=0.5)
        assert seg.end == pytest.approx(e, abs=0.5)

    out = tmp_path / "01 - Test.flac"
    job = TrackJob(side, segs[0].start, segs[0].end, out, {"TITLE": "Test", "TRACKNUMBER": "1"}, cover=b"\x89PNG fake", cover_mime="image/png")
    encode_track(job, cfg)
    f = FLAC(str(out))
    assert f["TITLE"] == ["Test"]
    assert f.info.bits_per_sample == 24
    assert f.info.length == pytest.approx(segs[0].length, abs=0.05)
    assert f.pictures and f.pictures[0].mime == "image/png"


def test_cli_rip_from_file(tmp_path: Path):
    sig, _ = build([15, 10])
    side_a = tmp_path / "a.wav"
    side_b = tmp_path / "b.wav"
    write_wav(side_a, sig)
    write_wav(side_b, sig)
    music = tmp_path / "music"
    answers = "\n".join([
        "Test Artist", "Test Album", "1979", "1",   # manual release
        "ok", "Opener", "Second",                   # side A: accept, titles
        "ok", "Third", "",                          # side B: accept, titles (last uses default)
    ]) + "\n"
    res = subprocess.run(
        [sys.executable, "-m", "vinylrip", "rip", "--no-metadata", "-o", str(music), "--from-file", str(side_a), str(side_b)],
        input=answers, capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    files = sorted((music / "Test Artist" / "Test Album (1979)").glob("*.flac"))
    assert [f.name for f in files] == ["01 - Opener.flac", "02 - Second.flac", "03 - Third.flac", "04 - Track 4.flac"]
    tags = FLAC(str(files[3]))
    assert tags["TRACKTOTAL"] == ["4"]
    assert tags["ALBUMARTIST"] == ["Test Artist"]
    assert tags["SIDE"] == ["B"]
