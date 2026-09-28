"""Audio input device discovery and ffmpeg input-argument construction."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class InputDevice:
    backend: str            # avfoundation | pulse | alsa
    identifier: str         # what ffmpeg's -i needs
    name: str               # human readable
    is_default: bool = False
    channels: int | None = None
    sample_rate: int | None = None
    extra: dict = field(default_factory=dict)

    def ffmpeg_input_args(self) -> list[str]:
        if self.backend == "portaudio":
            raise ValueError("portaudio devices are not captured through ffmpeg")
        if self.backend == "avfoundation":
            return ["-f", "avfoundation", "-i", f":{self.identifier}"]
        if self.backend == "pulse":
            return ["-f", "pulse", "-i", self.identifier]
        if self.backend == "alsa":
            return ["-f", "alsa", "-i", self.identifier]
        raise ValueError(f"unknown backend {self.backend}")

    def describe(self) -> str:
        bits = [self.name]
        if self.channels:
            bits.append(f"{self.channels} ch")
        if self.sample_rate:
            bits.append(f"{self.sample_rate} Hz")
        bits.append(f"{self.backend}:{self.identifier}")
        return " · ".join(bits)


class DeviceError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------- macOS

def _coreaudio_inputs() -> list[dict]:
    """Input devices from system_profiler (gives default flag, channels, rate)."""
    try:
        res = _run(["system_profiler", "SPAudioDataType", "-json"], timeout=30)
        data = json.loads(res.stdout)
    except Exception:
        return []
    out = []
    for group in data.get("SPAudioDataType", []):
        for item in group.get("_items", []):
            if "coreaudio_device_input" not in item:
                continue
            out.append(
                {
                    "name": item.get("_name", "?"),
                    "default": item.get("coreaudio_default_audio_input_device") == "spaudio_yes",
                    "channels": item.get("coreaudio_device_input"),
                    "sample_rate": item.get("coreaudio_device_srate"),
                }
            )
    return out


def _avfoundation_audio_devices() -> list[tuple[int, str]]:
    res = _run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""])
    text = res.stderr
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in text.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if "AVFoundation video devices" in line:
            in_audio = False
            continue
        if in_audio:
            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if m:
                devices.append((int(m.group(1)), m.group(2).strip()))
    return devices


def list_devices_macos() -> list[InputDevice]:
    core = {d["name"]: d for d in _coreaudio_inputs()}
    devices = []
    for idx, name in _avfoundation_audio_devices():
        info = core.get(name, {})
        devices.append(
            InputDevice(
                backend="avfoundation",
                identifier=str(idx),
                name=name,
                is_default=bool(info.get("default")),
                channels=info.get("channels"),
                sample_rate=info.get("sample_rate"),
            )
        )
    return devices


# --------------------------------------------------------------------------- Linux

def _pulse_sources() -> list[InputDevice]:
    if not shutil.which("pactl"):
        return []
    try:
        res = _run(["pactl", "list", "short", "sources"])
    except Exception:
        return []
    if res.returncode != 0:
        return []
    default_name = ""
    try:
        d = _run(["pactl", "get-default-source"])
        default_name = d.stdout.strip()
    except Exception:
        pass
    devices = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        if name.endswith(".monitor"):
            continue
        spec = parts[3] if len(parts) > 3 else ""
        ch = rate = None
        m = re.search(r"(\d+)ch\s+(\d+)Hz", spec)
        if m:
            ch, rate = int(m.group(1)), int(m.group(2))
        devices.append(
            InputDevice("pulse", name, name, is_default=(name == default_name), channels=ch, sample_rate=rate)
        )
    return devices


def _alsa_devices() -> list[InputDevice]:
    if not shutil.which("arecord"):
        return []
    try:
        res = _run(["arecord", "-l"])
    except Exception:
        return []
    devices = []
    for line in res.stdout.splitlines():
        m = re.match(r"card (\d+): (\S+) \[(.+?)\], device (\d+): (.+?) \[(.+?)\]", line)
        if m:
            card, _, cardname, dev, _, devname = m.groups()
            devices.append(InputDevice("alsa", f"hw:{card},{dev}", f"{cardname} - {devname}"))
    return devices


def list_devices_linux(backend: str = "auto") -> list[InputDevice]:
    devices: list[InputDevice] = []
    if backend in ("auto", "pulse"):
        devices += _pulse_sources()
    if backend in ("auto", "alsa") and (backend == "alsa" or not devices):
        devices += _alsa_devices()
    return devices


# --------------------------------------------------------------------------- PortAudio

def list_devices_portaudio() -> list[InputDevice]:
    try:
        import sounddevice as sd
    except Exception as exc:  # missing wheel / libportaudio
        raise DeviceError(f"sounddevice/PortAudio unavailable ({exc}); set [audio] backend = \"ffmpeg\"") from exc
    devices = []
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    for idx, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        devices.append(
            InputDevice(
                backend="portaudio",
                identifier=str(idx),
                name=d["name"],
                is_default=(idx == default_in),
                channels=int(d["max_input_channels"]),
                sample_rate=int(d.get("default_samplerate") or 0) or None,
                extra={"hostapi": sd.query_hostapis(d["hostapi"])["name"]},
            )
        )
    return devices


# --------------------------------------------------------------------------- common

def list_devices(backend: str = "auto") -> list[InputDevice]:
    if backend in ("auto", "portaudio"):
        try:
            return list_devices_portaudio()
        except DeviceError:
            if backend == "portaudio":
                raise
    if not shutil.which("ffmpeg"):
        raise DeviceError("ffmpeg not found on PATH")
    system = platform.system()
    if system == "Darwin":
        return list_devices_macos()
    if system == "Linux":
        return list_devices_linux("auto" if backend in ("auto", "ffmpeg") else backend)
    raise DeviceError(f"Unsupported platform: {system}")


def resolve_device(spec: str, backend: str = "auto") -> InputDevice:
    """Turn a config/CLI device spec into an InputDevice."""
    spec = (spec or "default").strip()
    system = platform.system()
    devices = list_devices(backend)

    if spec.lower() == "default":
        for d in devices:
            if d.is_default:
                return d
        if system == "Linux":
            be = "alsa" if backend == "alsa" else "pulse"
            if be == "pulse" and not shutil.which("pactl"):
                be = "alsa"
            return InputDevice(be, "default", "default")
        if devices:
            return devices[0]
        raise DeviceError("No audio input devices found")

    if system == "Darwin" or (devices and devices[0].backend == "portaudio"):
        if spec.isdigit():
            for d in devices:
                if d.identifier == spec:
                    return d
            raise DeviceError(f"No audio input device with index {spec}")
        matches = [d for d in devices if d.name.lower() == spec.lower()]
        if not matches:
            matches = [d for d in devices if spec.lower() in d.name.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            names = ", ".join(d.name for d in devices)
            raise DeviceError(f"No input device matching '{spec}'. Available: {names}")
        names = ", ".join(d.name for d in matches)
        raise DeviceError(f"Device spec '{spec}' is ambiguous: {names}")

    # Linux: exact match on identifier or name, otherwise pass through verbatim.
    for d in devices:
        if spec in (d.identifier, d.name):
            return d
    for d in devices:
        if spec.lower() in d.name.lower():
            return d
    be = backend if backend in ("pulse", "alsa") else ("alsa" if spec.startswith(("hw:", "plughw:")) else "pulse")
    return InputDevice(be, spec, spec)
