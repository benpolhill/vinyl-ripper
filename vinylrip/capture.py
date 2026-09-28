"""Recording a side (or analysing an existing file).

Capture backends:
  * portaudio (default): the sounddevice library. Bit-exact, dropout-free, all channels.
  * ffmpeg: avfoundation/pulse/alsa via a pipe. Kept as a fallback; on macOS ffmpeg's
    avfoundation input was measured dropping ~15% of samples, so avoid it there.

Every block of audio is written to a WAV (all captured channels) and reduced to an
RMS level envelope used for the live meter, end-of-side detection and track splitting.
"""

from __future__ import annotations

import os
import queue
import select
import subprocess
import sys
import termios
import threading
import time
import tty
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .config import Config
from .devices import InputDevice
from .split import DetectionParams, auto_threshold, detect_segments

BYTES_PER_SAMPLE = 4  # int32


@dataclass
class SideAudio:
    path: Path                  # file ffmpeg can read for cutting
    sample_rate: int
    duration: float
    block_s: float
    envelope: np.ndarray        # RMS dBFS per block (of the chosen channel pair)
    peaks: np.ndarray           # peak dBFS per block
    channels: int = 2           # channels in `path`
    pair: tuple[int, int] | None = None   # (left, right) channel indexes to use from `path`
    clipped_blocks: int = 0
    stopped_by: str = ""        # "auto" | "user" | "eof"
    extra: dict = field(default_factory=dict)


class CaptureError(RuntimeError):
    pass


def _db(x: float) -> float:
    return 20 * np.log10(x) if x > 0 else -120.0


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes from a raw pipe (short reads are normal), fewer only at EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ----------------------------------------------------------------------------- block sources

class _BlockSource:
    """Yields int32 arrays of shape (frames, channels); None at end of stream."""

    channels: int

    def get(self, timeout: float = 0.1):
        raise NotImplementedError

    @property
    def backlog(self) -> int:
        return 0

    def close(self) -> None:
        pass


class _PortAudioSource(_BlockSource):
    def __init__(self, device_index: int, channels: int, sample_rate: int, block_frames: int) -> None:
        import sounddevice as sd

        self.channels = channels
        self.queue: queue.Queue = queue.Queue()
        self.overflows = 0

        def callback(indata, frames, _time, status):
            if status and status.input_overflow:
                self.overflows += 1
            self.queue.put(indata.copy())

        try:
            self.stream = sd.InputStream(
                device=device_index, channels=channels, samplerate=sample_rate, dtype="int32",
                blocksize=block_frames, callback=callback,
            )
            self.stream.start()
        except Exception as exc:  # PortAudioError and friends
            raise CaptureError(f"Could not open input device: {exc}") from exc

    def get(self, timeout: float = 0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return ...

    @property
    def backlog(self) -> int:
        return self.queue.qsize()

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class _FfmpegSource(_BlockSource):
    """ffmpeg decoding a device or a file to raw s32le on a pipe, drained by a thread."""

    def __init__(self, input_args: list[str], channels: int, sample_rate: int, block_frames: int,
                 extra_filters: list[str] | None = None) -> None:
        self.channels = channels
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *input_args]
        if extra_filters:
            cmd += ["-af", ",".join(extra_filters)]
        cmd += ["-ar", str(sample_rate), "-ac", str(channels), "-f", "s32le", "-c:a", "pcm_s32le", "pipe:1"]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.block_bytes = block_frames * channels * BYTES_PER_SAMPLE
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.proc.stdout is not None
        while True:
            data = _read_exact(self.proc.stdout, self.block_bytes)
            if not data:
                self.queue.put(None)
                return
            usable = len(data) - len(data) % (self.channels * BYTES_PER_SAMPLE)
            self.queue.put(np.frombuffer(data[:usable], dtype="<i4").reshape(-1, self.channels))

    def get(self, timeout: float = 0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return ...

    @property
    def backlog(self) -> int:
        return self.queue.qsize()

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def error_text(self) -> str:
        try:
            return self.proc.stderr.read().decode(errors="ignore").strip() if self.proc.stderr else ""
        except Exception:
            return ""


def open_device_source(device: InputDevice, cfg: Config, block_frames: int) -> _BlockSource:
    if device.backend == "portaudio":
        channels = device.channels or 2
        return _PortAudioSource(int(device.identifier), channels, cfg.audio.sample_rate, block_frames)
    # ffmpeg device backends: we cannot know the channel count up front, so ask for stereo and
    # let the configured pair (or 0/1) be selected by ffmpeg itself.
    pair = cfg.audio.input_channels if isinstance(cfg.audio.input_channels, list) else [0, 1]
    return _FfmpegSource(device.ffmpeg_input_args(), 2, cfg.audio.sample_rate, block_frames,
                         extra_filters=[f"pan=stereo|c0=c{pair[0]}|c1=c{pair[1]}"])


# ----------------------------------------------------------------------------- keyboard

class _KeyWatcher:
    """Non-blocking single-key reader for a TTY. No-op when stdin is not a TTY."""

    def __init__(self) -> None:
        self.fd = None
        self.saved = None
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()

    def __enter__(self) -> "_KeyWatcher":
        if self.fd is not None:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.fd is not None and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def poll(self) -> str | None:
        if self.fd is None:
            return None
        r, _, _ = select.select([self.fd], [], [], 0)
        if r:
            data = os.read(self.fd, 64)
            return data.decode(errors="ignore")
        return None


# ----------------------------------------------------------------------------- level maths

class _Levels:
    """Per-channel RMS/peak for one block, plus the running choice of stereo pair."""

    def __init__(self, channels: int, configured: list[int] | str) -> None:
        self.channels = channels
        self.fixed: tuple[int, int] | None = None
        if isinstance(configured, list):
            l, r = configured
            if max(l, r) >= channels:
                raise CaptureError(f"input_channels {configured} but the device has only {channels} channel(s)")
            self.fixed = (l, r)
        self.energy = np.zeros(channels)  # running sum of mean-square per channel

    def measure(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = block.astype(np.float64) / 2147483648.0
        ms = np.mean(x * x, axis=0)
        peak = np.max(np.abs(x), axis=0) if x.size else np.zeros(self.channels)
        self.energy += ms
        return ms, peak

    def pair(self) -> tuple[int, int]:
        if self.fixed:
            return self.fixed
        if self.channels <= 2:
            return (0, min(1, self.channels - 1))
        # loudest adjacent pair so far
        sums = [self.energy[i] + self.energy[i + 1] for i in range(self.channels - 1)]
        i = int(np.argmax(sums))
        return (i, i + 1)


def _meter_panel(label: str, elapsed: float, rms_db: float, peak_db: float, status: str, n_tracks: int,
                 clipped: int, threshold_db: float, pair_text: str = "") -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    bar_width = 50
    lo = -60.0
    fill = int(bar_width * min(1.0, max(0.0, (rms_db - lo) / -lo)))
    peak_pos = int(bar_width * min(1.0, max(0.0, (peak_db - lo) / -lo)))
    bar = Text()
    for i in range(bar_width):
        ch = "█" if i < fill else ("|" if i == peak_pos else "·")
        db_here = lo + (i / bar_width) * -lo
        style = "green" if db_here < -12 else ("yellow" if db_here < -3 else "red")
        bar.append(ch, style=style if (i < fill or i == peak_pos) else "dim")
    table.add_row("Side", f"{label}    elapsed {_fmt_time(elapsed)}" + (f"    {pair_text}" if pair_text else ""))
    table.add_row("Level", bar)
    table.add_row("", f"rms {rms_db:6.1f} dBFS   peak {peak_db:6.1f} dBFS   silence < {threshold_db:.0f} dBFS")
    table.add_row("Status", status)
    table.add_row("Tracks", f"{n_tracks} detected so far" + (f"   [red]clipping in {clipped} blocks[/red]" if clipped else ""))
    table.add_row("", "[dim]Enter = stop the side · Ctrl-C = abort[/dim]")
    return table


# ----------------------------------------------------------------------------- recording

def record_side(device: InputDevice, cfg: Config, out_path: Path, label: str, console: Console,
                auto_stop: bool = True) -> SideAudio:
    """Record from `device` into `out_path` (32-bit WAV) until silence or keypress."""
    det = cfg.detection
    block_s = det.block_ms / 1000.0
    block_frames = int(cfg.audio.sample_rate * block_s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = open_device_source(device, cfg, block_frames)
    nch = source.channels
    levels = _Levels(nch, cfg.audio.input_channels)

    wav = wave.open(str(out_path), "wb")
    wav.setnchannels(nch)
    wav.setsampwidth(BYTES_PER_SAMPLE)
    wav.setframerate(cfg.audio.sample_rate)

    envelope: list[float] = []
    peaks: list[float] = []
    clipped = 0
    stopped_by = "eof"
    music_seen = False
    silence_blocks = 0
    current_run = 0
    min_sound_blocks = max(1, round(det.min_sound_seconds / block_s))
    track_blocks = max(1, round(det.min_track_seconds / block_s))
    end_blocks = max(1, round(det.side_end_silence_seconds / block_s))
    grace_blocks = max(0, round(det.side_end_grace_seconds / block_s))
    fixed_thr = None if det.silence_threshold_db == "auto" else float(det.silence_threshold_db)
    threshold = fixed_thr if fixed_thr is not None else -40.0
    n_tracks = 0
    last_ui = 0.0
    last_detect = 0.0
    status = "starting capture…"
    rms_db = peak_db = -120.0

    try:
        with _KeyWatcher() as keys, Live(console=console, refresh_per_second=10, transient=True) as live:
            live.update(_meter_panel(label, 0, rms_db, peak_db, status, 0, 0, threshold))
            while True:
                key = keys.poll()
                if key is not None:
                    if "c" in key.lower() and music_seen and silence_blocks >= end_blocks:
                        music_seen = False          # quiet passage: re-arm and keep going
                        silence_blocks = 0
                        status = "[green]recording[/green] (auto-stop re-armed)"
                    else:
                        stopped_by = "user"
                        break
                block = source.get()
                if block is ...:
                    continue
                if block is None:
                    break
                wav.writeframesraw(block.astype("<i4").tobytes())
                ms, pk = levels.measure(block)
                l, r = levels.pair()
                rms_db = _db(float(np.sqrt((ms[l] + ms[r]) / 2)))
                peak = float(max(pk[l], pk[r]))
                peak_db = _db(peak)
                if peak >= 0.999:
                    clipped += 1
                envelope.append(rms_db)
                peaks.append(peak_db)
                elapsed = len(envelope) * block_s
                if len(envelope) == 1:
                    status = "[bold green]capturing – drop the needle now[/bold green]"

                if fixed_thr is None and len(envelope) % 200 == 0 and elapsed >= 30:
                    threshold = auto_threshold(np.asarray(envelope), det.auto_margin_db)

                if rms_db > threshold:
                    current_run += 1
                    if current_run >= min_sound_blocks:
                        silence_blocks = 0
                    if current_run >= track_blocks:
                        music_seen = True
                else:
                    current_run = 0
                    silence_blocks += 1

                if music_seen:
                    if silence_blocks >= end_blocks and auto_stop:
                        left = (end_blocks + grace_blocks - silence_blocks) * block_s
                        status = (f"[yellow]side seems finished[/yellow] – stopping in {left:.0f}s "
                                  f"([Enter] stop now, [c] keep recording)")
                        if silence_blocks >= end_blocks + grace_blocks:
                            stopped_by = "auto"
                            break
                    elif silence_blocks:
                        status = f"silence {silence_blocks * block_s:.1f}s / {det.side_end_silence_seconds:.0f}s"
                    else:
                        status = "[green]recording[/green]"

                now = time.monotonic()
                if now - last_detect > 2.0:
                    last_detect = now
                    n_tracks = len(detect_segments(np.asarray(envelope), block_s, DetectionParams.from_config(det, threshold)))
                if now - last_ui > 0.1 and source.backlog < 2:
                    last_ui = now
                    pair_text = f"channels {l + 1}/{r + 1} of {nch}" if nch > 2 else ""
                    live.update(_meter_panel(label, elapsed, rms_db, peak_db, status, n_tracks, clipped, threshold, pair_text))
    finally:
        source.close()
        wav.close()

    if not envelope:
        detail = source.error_text() if isinstance(source, _FfmpegSource) else ""
        raise CaptureError(f"No audio received from the input device. {detail}".strip())

    return SideAudio(
        path=out_path,
        sample_rate=cfg.audio.sample_rate,
        duration=len(envelope) * block_s,
        block_s=block_s,
        envelope=np.asarray(envelope),
        peaks=np.asarray(peaks),
        channels=nch,
        pair=levels.pair(),
        clipped_blocks=clipped,
        stopped_by=stopped_by,
        extra={"overflows": getattr(source, "overflows", 0)},
    )


def analyze_file(path: Path, cfg: Config, console: Console | None = None) -> SideAudio:
    """Compute the level envelope of an existing audio file (no re-encoding)."""
    det = cfg.detection
    block_s = det.block_ms / 1000.0
    sr, nch = probe_audio(path)
    sr = sr or cfg.audio.sample_rate
    nch = nch or 2
    block_frames = int(sr * block_s)
    source = _FfmpegSource(["-i", str(path)], nch, sr, block_frames)
    levels = _Levels(nch, cfg.audio.input_channels if nch > 2 else [0, min(1, nch - 1)])
    envelope: list[float] = []
    peaks: list[float] = []
    clipped = 0
    try:
        while True:
            block = source.get(timeout=5.0)
            if block is ...:
                continue
            if block is None:
                break
            ms, pk = levels.measure(block)
            l, r = levels.pair()
            rms = float(np.sqrt((ms[l] + ms[r]) / 2))
            peak = float(max(pk[l], pk[r]))
            if peak >= 0.999:
                clipped += 1
            envelope.append(_db(rms))
            peaks.append(_db(peak))
    finally:
        source.close()
    if not envelope:
        raise CaptureError(f"ffmpeg could not read {path}: {source.error_text()}")
    return SideAudio(
        path=path,
        sample_rate=sr,
        duration=len(envelope) * block_s,
        block_s=block_s,
        envelope=np.asarray(envelope),
        peaks=np.asarray(peaks),
        channels=nch,
        pair=levels.pair() if nch > 2 else None,
        clipped_blocks=clipped,
        stopped_by="eof",
    )


def probe_audio(path: Path) -> tuple[int | None, int | None]:
    """(sample_rate, channels) of the first audio stream, via ffprobe."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        sr, ch = (int(x) for x in res.stdout.split())
        return sr, ch
    except Exception:
        return None, None


def monitor_levels(device: InputDevice, cfg: Config, console: Console) -> None:
    """Show a live meter without recording. Useful for setting gain and checking channels."""
    block_s = cfg.detection.block_ms / 1000.0
    block_frames = int(cfg.audio.sample_rate * block_s)
    source = open_device_source(device, cfg, block_frames)
    nch = source.channels
    levels = _Levels(nch, cfg.audio.input_channels)
    env: list[float] = []
    clipped = 0
    peak_hold = -120.0
    last_ui = 0.0
    per_channel = np.zeros(nch)
    try:
        with _KeyWatcher() as keys, Live(console=console, refresh_per_second=10, transient=True) as live:
            while True:
                if keys.poll() is not None:
                    break
                block = source.get()
                if block is ...:
                    continue
                if block is None:
                    break
                ms, pk = levels.measure(block)
                per_channel = 0.9 * per_channel + 0.1 * ms
                l, r = levels.pair()
                rms_db = _db(float(np.sqrt((ms[l] + ms[r]) / 2)))
                peak = float(max(pk[l], pk[r]))
                peak_db = _db(peak)
                peak_hold = max(peak_hold, peak_db)
                if peak >= 0.999:
                    clipped += 1
                env.append(rms_db)
                now = time.monotonic()
                if now - last_ui > 0.1 and source.backlog < 2:
                    last_ui = now
                    floor = float(np.percentile(env[-1200:], 10)) if len(env) > 20 else rms_db
                    chans = "  ".join(f"ch{i + 1} {_db(float(np.sqrt(v))):6.1f}" for i, v in enumerate(per_channel))
                    status = f"noise floor ≈ {floor:.1f} dBFS   peak hold {peak_hold:.1f} dBFS\n{chans}"
                    pair_text = f"using channels {l + 1}/{r + 1} of {nch}" if nch > 2 else ""
                    live.update(_meter_panel("levels", len(env) * block_s, rms_db, peak_db, status, 0, clipped,
                                             floor + cfg.detection.auto_margin_db, pair_text))
    finally:
        source.close()
    if not env:
        raise CaptureError("No audio received from the input device.")
