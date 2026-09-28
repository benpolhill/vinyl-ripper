"""Cutting tracks out of a side recording, encoding to FLAC and tagging."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mutagen.flac import FLAC, Picture

from .config import Config
from .metadata import Release, Track


class EncodeError(RuntimeError):
    pass


@dataclass
class TrackJob:
    source: Path
    start: float
    end: float
    out_path: Path
    tags: dict[str, str]
    cover: bytes | None = None
    cover_mime: str = "image/jpeg"
    pair: tuple[int, int] | None = None   # channels of `source` to use as L/R


def sanitize(name: str, max_len: int = 120) -> str:
    name = re.sub(r"[\x00-\x1f/\\:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name or "untitled")[:max_len]


def album_dir(cfg: Config, rel: Release) -> Path:
    fields = {
        "albumartist": sanitize(rel.artist or "Unknown Artist"),
        "album": sanitize(rel.title or "Unknown Album"),
        "year": sanitize(rel.year or "unknown year"),
        "label": sanitize(rel.label),
        "catno": sanitize(rel.catno),
    }
    sub = cfg.paths.album_dir_template.format(**fields)
    return cfg.paths.music_dir / Path(*[sanitize(p) for p in Path(sub).parts])


def track_filename(cfg: Config, rel: Release, track: Track, disc_no: int, side_label: str) -> str:
    fields = {
        "disc": disc_no,
        "disc_prefix": f"{disc_no}-" if len(rel.discs) > 1 else "",
        "track": track.number,
        "title": sanitize(track.title or f"Track {track.number}"),
        "artist": sanitize(track.artist or rel.artist),
        "side": side_label,
        "position": sanitize(track.position),
        "album": sanitize(rel.title),
    }
    return sanitize(cfg.paths.track_template.format(**fields), max_len=200)


def build_tags(rel: Release, track: Track, disc_no: int, side_label: str) -> dict[str, str]:
    disc = next((d for d in rel.discs if d.number == disc_no), None)
    tags = {
        "TITLE": track.title or f"Track {track.number}",
        "ARTIST": track.artist or rel.artist,
        "ALBUMARTIST": rel.artist,
        "ALBUM": rel.title,
        "TRACKNUMBER": str(track.number),
        "TRACKTOTAL": str(len(disc.tracks)) if disc else "",
        "DISCNUMBER": str(disc_no),
        "DISCTOTAL": str(len(rel.discs)),
        "DATE": rel.year,
        "LABEL": rel.label,
        "CATALOGNUMBER": rel.catno,
        "RELEASECOUNTRY": rel.country,
        "MEDIA": "Vinyl",
        "SIDE": side_label,
        "MUSICBRAINZ_ALBUMID": rel.mb_release_id,
        "MUSICBRAINZ_TRACKID": track.mb_recording_id,
        "MUSICBRAINZ_RELEASETRACKID": track.mb_track_id,
        "DISCOGS_RELEASE_ID": rel.discogs_id,
        "ENCODER": "vinylrip",
    }
    return {k: v for k, v in tags.items() if v}


def encode_track(job: TrackJob, cfg: Config) -> Path:
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    length = job.end - job.start
    fade = cfg.detection.fade_ms / 1000.0
    filters = []
    if job.pair is not None:
        filters.append(f"pan=stereo|c0=c{job.pair[0]}|c1=c{job.pair[1]}")
    if fade > 0 and length > 2 * fade:
        filters.append(f"afade=t=in:st=0:d={fade:.4f}")
        filters.append(f"afade=t=out:st={length - fade:.4f}:d={fade:.4f}")
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{job.start:.4f}", "-t", f"{length:.4f}", "-i", str(job.source),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    sample_fmt = "s16" if cfg.audio.bit_depth == 16 else "s32"
    cmd += ["-c:a", "flac", "-sample_fmt", sample_fmt, "-compression_level", str(cfg.encode.compression_level)]
    if cfg.audio.bit_depth == 24:
        cmd += ["-bits_per_raw_sample", "24"]
    cmd += ["-map_metadata", "-1", str(job.out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise EncodeError(f"ffmpeg failed for {job.out_path.name}: {res.stderr.strip()}")
    tag_flac(job.out_path, job.tags, job.cover, job.cover_mime)
    return job.out_path


def tag_flac(path: Path, tags: dict[str, str], cover: bytes | None, cover_mime: str = "image/jpeg") -> None:
    audio = FLAC(str(path))
    audio.delete()
    for k, v in tags.items():
        audio[k] = v
    audio.clear_pictures()
    if cover:
        pic = Picture()
        pic.type = 3  # front cover
        pic.mime = cover_mime or "image/jpeg"
        pic.desc = "Front cover"
        pic.data = cover
        audio.add_picture(pic)
    audio.save()
