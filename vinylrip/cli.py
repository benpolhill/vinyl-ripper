"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .capture import CaptureError, analyze_file, monitor_levels
from .config import Config, config_path, load_config, write_default_config
from .devices import DeviceError, list_devices, resolve_device
from .rip import Abort, RipOptions, fmt_time, run_rip
from . import split as sp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vinylrip", description="Rip vinyl records to tagged FLAC from the command line.")
    p.add_argument("--version", action="version", version=f"vinylrip {__version__}")
    p.add_argument("--config", type=Path, help="config file (default: ~/.config/vinylrip/config.toml)")
    sub = p.add_subparsers(dest="command")

    rip = sub.add_parser("rip", help="record a record side by side and save tagged FLAC tracks")
    rip.add_argument("-a", "--artist", default="", help="artist to search for")
    rip.add_argument("-r", "--album", "--release", default="", help="album title to search for")
    rip.add_argument("-d", "--device", help="input device (overrides config)")
    rip.add_argument("--rate", type=int, help="sample rate (overrides config)")
    rip.add_argument("--bits", type=int, choices=(16, 24), help="FLAC bit depth (overrides config)")
    rip.add_argument("-o", "--music-dir", type=Path, help="destination folder (overrides config)")
    rip.add_argument("--no-metadata", action="store_true", help="skip online lookup, enter details manually")
    rip.add_argument("--no-auto-stop", action="store_true", help="never stop a side on silence; stop with Enter")
    rip.add_argument("--threshold", type=float, help="silence threshold in dBFS (overrides auto detection)")
    rip.add_argument("--from-file", nargs="+", type=Path, metavar="SIDE", help="process existing side recordings instead of capturing")
    rip.add_argument("--sides", help="comma-separated side labels for --from-file inputs, e.g. C,D,A,B")

    sub.add_parser("devices", help="list audio input devices")

    lv = sub.add_parser("levels", help="show a live input meter (for setting gain); press Enter to stop")
    lv.add_argument("-d", "--device", help="input device (overrides config)")

    det = sub.add_parser("detect", help="show the tracks that would be detected in an existing recording")
    det.add_argument("file", type=Path)
    det.add_argument("--threshold", type=float, help="silence threshold in dBFS")

    cfg = sub.add_parser("config", help="manage the config file")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    ini = cfg_sub.add_parser("init", help="write a default config file")
    ini.add_argument("--force", action="store_true", help="overwrite an existing file")
    cfg_sub.add_parser("show", help="print the effective configuration")
    cfg_sub.add_parser("path", help="print the config file location")
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if getattr(args, "device", None):
        cfg.audio.device = args.device
    if getattr(args, "rate", None):
        cfg.audio.sample_rate = args.rate
    if getattr(args, "bits", None):
        cfg.audio.bit_depth = args.bits
    if getattr(args, "music_dir", None):
        cfg.paths.music_dir = args.music_dir.expanduser()
    if getattr(args, "no_metadata", False):
        cfg.metadata.enabled = False
    if getattr(args, "threshold", None) is not None:
        cfg.detection.silence_threshold_db = args.threshold


def cmd_devices(cfg: Config, console: Console) -> int:
    devices = list_devices(cfg.audio.backend)
    if not devices:
        console.print("[yellow]No input devices found.[/yellow]")
        return 1
    table = Table(title="Audio input devices")
    table.add_column("")
    table.add_column("Name")
    table.add_column("Channels", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Backend id")
    for d in devices:
        table.add_row("*" if d.is_default else "", d.name, str(d.channels or ""), str(d.sample_rate or ""), f"{d.backend}:{d.identifier}")
    console.print(table)
    console.print("[dim]* = system default input. Configure with \\[audio] device = \"<name or id>\".[/dim]")
    return 0


def cmd_detect(cfg: Config, args: argparse.Namespace, console: Console) -> int:
    with console.status(f"Analysing {args.file.name}..."):
        audio = analyze_file(args.file, cfg, console)
    thr = sp.resolve_threshold(audio.envelope, cfg.detection)
    params = sp.DetectionParams.from_config(cfg.detection, thr)
    segments = sp.detect_segments(audio.envelope, audio.block_s, params)
    console.print(f"{args.file}: {fmt_time(audio.duration).strip()}, noise floor {sp.noise_floor_db(audio.envelope):.1f} dBFS, "
                  f"threshold {thr:.1f} dBFS, clipped blocks {audio.clipped_blocks}")
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Length", justify="right")
    for i, s in enumerate(segments, start=1):
        table.add_row(str(i), fmt_time(s.start), fmt_time(s.end), fmt_time(s.length))
    console.print(table)
    return 0


def cmd_config(args: argparse.Namespace, console: Console) -> int:
    path = args.config or config_path()
    if args.config_command == "path":
        print(path)
        return 0
    if args.config_command == "init":
        try:
            written = write_default_config(path, force=args.force)
        except FileExistsError:
            console.print(f"[yellow]{path} already exists (use --force to overwrite).[/yellow]")
            return 1
        console.print(f"Wrote default config to {written}")
        return 0
    cfg = load_config(path)
    console.print(f"[dim]config file: {cfg.source or f'{path} (not found, using defaults)'}[/dim]")
    for section in ("paths", "audio", "detection", "metadata", "encode"):
        console.print(f"[bold]\\[{section}][/bold]")
        for k, v in vars(getattr(cfg, section)).items():
            if k == "discogs_token" and v:
                v = "****"
            console.print(f"  {k} = {v!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "config":
        if args.config_command is None:
            parser.parse_args(["config", "--help"])
            return 0
        return cmd_config(args, console)

    try:
        cfg = load_config(args.config)
    except (ValueError, OSError) as exc:
        console.print(f"[red]Config error: {exc}[/red]")
        return 2
    apply_overrides(cfg, args)

    try:
        if args.command == "devices":
            return cmd_devices(cfg, console)
        if args.command == "detect":
            return cmd_detect(cfg, args, console)
        if args.command == "levels":
            device = resolve_device(cfg.audio.device, cfg.audio.backend)
            console.print(f"[bold]Input:[/bold] {device.describe()}")
            monitor_levels(device, cfg, console)
            return 0
        if args.command == "rip":
            device = None
            if not args.from_file:
                device = resolve_device(cfg.audio.device, cfg.audio.backend)
            for f in args.from_file or []:
                if not f.exists():
                    console.print(f"[red]File not found: {f}[/red]")
                    return 2
            sides = [x.strip() for x in args.sides.split(",")] if args.sides else None
            opts = RipOptions(artist=args.artist, album=args.album, files=args.from_file, sides=sides, no_auto_stop=args.no_auto_stop)
            files = run_rip(cfg, device, opts, console)
            return 0 if files else 1
    except Abort:
        console.print("[yellow]Aborted.[/yellow]")
        return 130
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted.[/yellow]")
        return 130
    except (DeviceError, CaptureError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
