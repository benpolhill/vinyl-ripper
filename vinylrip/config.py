"""Configuration handling: a TOML file with sensible defaults.

Location (first that applies):
  * $VINYLRIP_CONFIG
  * $XDG_CONFIG_HOME/vinylrip/config.toml
  * ~/.config/vinylrip/config.toml
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_TEXT = """\
# vinylrip configuration
# Created by `vinylrip config init`. All keys are optional; missing keys use the defaults shown.

[paths]
# Where finished, tagged FLAC files go. Album folders are created underneath.
music_dir = "~/Music/Vinyl"
# Where raw, untouched side recordings (32-bit WAV) are kept while ripping.
raw_dir = "~/Music/Vinyl/raw"
# Keep the raw side recordings after a successful rip (true) or delete them (false).
keep_raw = true
# Folder layout under music_dir. Fields: albumartist, album, year, label, catno
album_dir_template = "{albumartist}/{album} ({year})"
# File name per track. Fields: disc, track, title, artist, side, position, album
# {disc_prefix} expands to "1-" style prefix only for multi-disc releases.
track_template = "{disc_prefix}{track:02d} - {title}.flac"

[audio]
# Input device. "default" = the system default input.
# macOS: a (partial, case-insensitive) device name or an avfoundation index number.
# Linux: a PulseAudio/PipeWire source name, an ALSA device (e.g. "hw:1,0") or "default".
device = "default"
# Capture backend: "auto" (PortAudio via the sounddevice library, falling back to ffmpeg),
# "portaudio", or "ffmpeg" (avfoundation/pulse/alsa). On Linux "pulse"/"alsa" force that ffmpeg input.
backend = "auto"
sample_rate = 44100
# Bit depth of the saved FLAC files: 16 or 24.
bit_depth = 24
# Which two input channels are left/right: "auto" picks the loudest adjacent pair while
# recording (handy for multi-channel interfaces), or give them explicitly, e.g. [2, 3].
input_channels = "auto"

[detection]
# RMS level (dBFS) below which audio counts as silence, or "auto" to derive it from the
# recording's own noise floor (recommended). Typical manual values: -45 to -35.
silence_threshold_db = "auto"
# In auto mode the threshold is the noise floor plus this many dB.
auto_margin_db = 15.0
# Silence at least this long separates two tracks.
min_gap_seconds = 1.5
# Anything shorter than this is not a track (rejects clicks, pops and lock-groove ticks).
min_track_seconds = 5.0
# Bursts of sound shorter than this are ignored entirely (needle drops, clicks).
min_sound_seconds = 0.3
# A burst at the very start/end of a track shorter than this, separated from the music
# by at least pop_gap_seconds of silence, is trimmed off as a pop.
pop_max_seconds = 1.0
pop_gap_seconds = 0.4
# Audio kept before/after each detected track so quiet intros and decays are not cut.
pad_seconds = 0.25
# Fade applied at each cut (milliseconds) to avoid clicks.
fade_ms = 15
# While recording, stop the side automatically after this much silence once music has played.
side_end_silence_seconds = 15.0
# ...but keep capturing for this long first, so you can press [c] to continue if it was
# just a quiet passage. Unattended, the side stops after silence + grace.
side_end_grace_seconds = 30.0
# Envelope resolution used for detection.
block_ms = 50

[metadata]
# Set to false to skip online lookups entirely.
enabled = true
# MusicBrainz asks for a contact address in the User-Agent (email or URL).
musicbrainz_contact = ""
# Discogs personal access token (https://www.discogs.com/settings/developers).
# Optional: search works without one; a token raises the rate limit and enables Discogs cover art.
discogs_token = ""
# Embed front cover art from the Cover Art Archive / Discogs when available.
embed_cover = true

[encode]
# FLAC compression level 0-8.
compression_level = 8
"""


@dataclass
class PathsConfig:
    music_dir: Path = Path("~/Music/Vinyl").expanduser()
    raw_dir: Path = Path("~/Music/Vinyl/raw").expanduser()
    keep_raw: bool = True
    album_dir_template: str = "{albumartist}/{album} ({year})"
    track_template: str = "{disc_prefix}{track:02d} - {title}.flac"


@dataclass
class AudioConfig:
    device: str = "default"
    backend: str = "auto"
    sample_rate: int = 44100
    bit_depth: int = 24
    input_channels: list[int] | str = "auto"


@dataclass
class DetectionConfig:
    silence_threshold_db: float | str = "auto"
    auto_margin_db: float = 15.0
    min_gap_seconds: float = 1.5
    min_track_seconds: float = 5.0
    min_sound_seconds: float = 0.3
    pop_max_seconds: float = 1.0
    pop_gap_seconds: float = 0.4
    pad_seconds: float = 0.25
    fade_ms: int = 15
    side_end_silence_seconds: float = 15.0
    side_end_grace_seconds: float = 30.0
    block_ms: int = 50


@dataclass
class MetadataConfig:
    enabled: bool = True
    musicbrainz_contact: str = ""
    discogs_token: str = ""
    embed_cover: bool = True


@dataclass
class EncodeConfig:
    compression_level: int = 8


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    source: Path | None = None  # file the config was loaded from, if any

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> "Config":
        cfg = cls(source=source)
        for section_name, section in (
            ("paths", cfg.paths),
            ("audio", cfg.audio),
            ("detection", cfg.detection),
            ("metadata", cfg.metadata),
            ("encode", cfg.encode),
        ):
            values = data.get(section_name, {})
            if not isinstance(values, dict):
                raise ValueError(f"[{section_name}] must be a table")
            for key, value in values.items():
                if not hasattr(section, key):
                    raise ValueError(f"Unknown config key [{section_name}] {key}")
                setattr(section, key, value)
        cfg.paths.music_dir = Path(str(cfg.paths.music_dir)).expanduser()
        cfg.paths.raw_dir = Path(str(cfg.paths.raw_dir)).expanduser()
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.audio.bit_depth not in (16, 24):
            raise ValueError("audio.bit_depth must be 16 or 24")
        ch = self.audio.input_channels
        if not (ch == "auto" or (isinstance(ch, list) and len(ch) == 2 and all(isinstance(c, int) and c >= 0 for c in ch))):
            raise ValueError('audio.input_channels must be "auto" or two channel indexes, e.g. [0, 1]')
        thr = self.detection.silence_threshold_db
        if not (thr == "auto" or isinstance(thr, (int, float))):
            raise ValueError('detection.silence_threshold_db must be a number or "auto"')
        if self.audio.backend not in ("auto", "portaudio", "ffmpeg", "pulse", "alsa", "avfoundation"):
            raise ValueError("audio.backend must be auto, portaudio, ffmpeg, pulse, alsa or avfoundation")
        if not 0 <= self.encode.compression_level <= 8:
            raise ValueError("encode.compression_level must be 0-8")


def config_path() -> Path:
    env = os.environ.get("VINYLRIP_CONFIG")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "vinylrip" / "config.toml"


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return Config.from_dict(data, source=path)


def write_default_config(path: Path | None = None, force: bool = False) -> Path:
    path = path or config_path()
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEXT)
    return path
