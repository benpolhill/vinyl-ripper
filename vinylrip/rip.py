"""The interactive rip workflow: metadata -> record sides -> review splits -> encode."""

from __future__ import annotations

import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import split as sp
from .capture import SideAudio, analyze_file, record_side
from .config import Config
from .devices import InputDevice
from .encode import TrackJob, album_dir, build_tags, encode_track, sanitize, tag_flac, track_filename
from .metadata import Candidate, MetadataLookup, Release, Side, Track, manual_release


class Abort(Exception):
    """User asked to quit."""


# ----------------------------------------------------------------------------- prompts

def ask(console: Console, prompt: str, default: str = "") -> str:
    suffix = f" [dim]({default})[/dim]" if default else ""
    try:
        answer = console.input(f"[bold cyan]?[/] {prompt}{suffix}: ").strip()
    except EOFError:
        raise Abort()
    return answer or default


def ask_int(console: Console, prompt: str, default: int) -> int:
    while True:
        raw = ask(console, prompt, str(default))
        if raw.isdigit():
            return int(raw)
        console.print("[red]Please enter a whole number.[/red]")


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "  --:--"
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):3d}:{s:04.1f}"


def parse_time(text: str) -> float:
    """'3:45', '1:02:03', '45', '3:45.5' -> seconds."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"bad time '{text}' (use mm:ss)")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


# ----------------------------------------------------------------------------- metadata step

def choose_release(console: Console, cfg: Config, artist: str, album: str) -> Release:
    if not cfg.metadata.enabled:
        console.print("[yellow]Metadata lookup disabled in config; entering details manually.[/yellow]")
        return prompt_manual_release(console, artist, album)

    lookup = MetadataLookup(cfg.metadata)
    while True:
        if not artist and not album:
            artist = ask(console, "Artist (leave both blank to enter details manually)")
            album = ask(console, "Album")
            if not artist and not album:
                return prompt_manual_release(console, "", "")
        with console.status(f"Searching MusicBrainz{' and Discogs' if lookup.discogs.available else ''}..."):
            results, warnings = lookup.search(artist, album)
        for w in warnings:
            console.print(f"[dim]{w}[/dim]")
        if not results:
            console.print(f"[yellow]No releases found for '{artist}' / '{album}'.[/yellow]")
        else:
            _print_candidates(console, results)
        choice = ask(console, "Pick a number, [s]earch again, [m]anual entry, [q]uit", "1" if results else "s")
        if choice.lower() == "q":
            raise Abort()
        if choice.lower() == "s":
            artist = ask(console, "Artist", artist)
            album = ask(console, "Album", album)
            continue
        if choice.lower() == "m":
            return prompt_manual_release(console, artist, album)
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            cand = results[int(choice) - 1]
            with console.status(f"Fetching {cand.source} release..."):
                try:
                    rel = lookup.fetch(cand)
                except Exception as exc:  # network / parse errors
                    console.print(f"[red]Could not fetch release: {exc}[/red]")
                    continue
            print_release(console, rel)
            if ask(console, "Use this release? [Y/n]", "y").lower().startswith("y"):
                return rel
            continue
        console.print("[red]Not a valid choice.[/red]")


def _print_candidates(console: Console, results: list[Candidate]) -> None:
    table = Table(title="Matching releases", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Src")
    table.add_column("Artist", max_width=28, no_wrap=True, overflow="ellipsis")
    table.add_column("Title", max_width=30, no_wrap=True, overflow="ellipsis")
    table.add_column("Year")
    table.add_column("Country")
    table.add_column("Label / Cat#", max_width=26, no_wrap=True, overflow="ellipsis")
    table.add_column("Format", max_width=16)
    table.add_column("Tracks", justify="right")
    for i, c in enumerate(results, start=1):
        table.add_row(
            str(i), "MB" if c.source == "musicbrainz" else "DC", c.artist, c.title, c.year, c.country,
            " ".join(x for x in (c.label, c.catno) if x), c.format, str(c.track_count or ""),
        )
    console.print(table)


def print_release(console: Console, rel: Release) -> None:
    head = f"[bold]{rel.artist} – {rel.title}[/bold]"
    bits = [b for b in (rel.year, rel.country, rel.label, rel.catno) if b]
    if bits:
        head += "  [dim]" + " · ".join(bits) + "[/dim]"
    console.print(head)
    sides = rel.sides()
    if not sides:
        console.print("[dim]  (no track list; titles will be entered per side)[/dim]")
        return
    for side in sides:
        label = f"Side {side.label}" if side.known else "Track list"
        if len(rel.discs) > 1:
            label = f"Disc {side.disc} · {label}"
        console.print(f"  [cyan]{label}[/cyan]")
        for t in side.tracks:
            console.print(f"    {t.position:>4} {fmt_time(t.length_s)}  {t.title}")
    if rel.cover:
        console.print(f"  [dim]cover art: {len(rel.cover) // 1024} kB[/dim]")


def prompt_manual_release(console: Console, artist: str, album: str) -> Release:
    console.print("[bold]Enter release details[/bold] (track titles are asked for after each side)")
    artist = ask(console, "Album artist", artist)
    album = ask(console, "Album title", album)
    year = ask(console, "Year", "")
    discs = ask_int(console, "Number of discs", 1)
    return manual_release(artist, album, year, discs)


# ----------------------------------------------------------------------------- segment review

def _print_segments(console: Console, label: str, audio: SideAudio, segments: list[sp.Segment],
                    expected: list[Track] | None, threshold: float) -> None:
    table = Table(title=f"Side {label}: {len(segments)} track(s) detected in {fmt_time(audio.duration).strip()}"
                        f"  (silence < {threshold:.0f} dBFS)")
    table.add_column("#", justify="right")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Length", justify="right")
    if expected is not None:
        table.add_column("Expected track")
        table.add_column("Exp. length", justify="right")
    rows = max(len(segments), len(expected or []))
    for i in range(rows):
        cells = [str(i + 1)]
        if i < len(segments):
            s = segments[i]
            cells += [fmt_time(s.start), fmt_time(s.end), fmt_time(s.length)]
        else:
            cells += ["[red]—[/red]", "", ""]
        if expected is not None:
            if i < len(expected):
                t = expected[i]
                cells.append(f"{t.position} {t.title}".strip())
                exp = fmt_time(t.length_s)
                if t.length_s and i < len(segments):
                    diff = segments[i].length - t.length_s
                    colour = "green" if abs(diff) < 8 else ("yellow" if abs(diff) < 25 else "red")
                    exp = f"[{colour}]{exp} ({diff:+.0f}s)[/{colour}]"
                cells.append(exp)
            else:
                cells += ["[yellow](no expected track)[/yellow]", ""]
        table.add_row(*cells)
    console.print(table)
    if audio.clipped_blocks:
        console.print(f"[red]Warning: clipping detected in {audio.clipped_blocks} blocks – consider lowering the input gain.[/red]")


def review_segments(console: Console, cfg: Config, label: str, audio: SideAudio,
                    expected: list[Track] | None, allow_redo: bool) -> list[sp.Segment] | None:
    """Interactive loop; returns accepted segments, or None to re-record the side."""
    det = cfg.detection
    threshold = sp.resolve_threshold(audio.envelope, det)
    params = sp.DetectionParams.from_config(det, threshold)
    segments = sp.detect_segments(audio.envelope, audio.block_s, params)
    console.print(f"[dim]noise floor ≈ {sp.noise_floor_db(audio.envelope):.1f} dBFS[/dim]")

    while True:
        _print_segments(console, label, audio, segments, expected, params.threshold_db)
        if expected is not None and len(segments) != len(expected):
            console.print(f"[yellow]Detected {len(segments)} track(s) but {len(expected)} expected on this side.[/yellow]")
        console.print(
            "[dim]Commands: [Enter]/ok accept · m N merge N with next · s N mm:ss split N at time · d N delete · "
            "a mm:ss mm:ss add · e N mm:ss mm:ss set bounds · t dB threshold · g sec min gap · "
            "l sec min track length" + (" · r re-record side" if allow_redo else "") + " · q quit[/dim]"
        )
        raw = ask(console, "Edit", "ok")
        cmd, *args = raw.split()
        cmd = cmd.lower()
        try:
            if cmd in ("ok", "y", "yes", "a") and not args:
                if not segments:
                    console.print("[red]No tracks to save. Add one, lower the threshold or re-record.[/red]")
                    continue
                return segments
            if cmd == "q":
                raise Abort()
            if cmd == "r" and allow_redo:
                return None
            if cmd == "m":
                segments = sp.merge(segments, int(args[0]) - 1)
            elif cmd == "s":
                segments = sp.split(segments, int(args[0]) - 1, parse_time(args[1]))
            elif cmd == "d":
                segments = sp.delete(segments, int(args[0]) - 1)
            elif cmd == "a":
                segments = sp.add(segments, parse_time(args[0]), parse_time(args[1]))
            elif cmd == "e":
                start = None if args[1] in ("-", ".") else parse_time(args[1])
                end = None if len(args) < 3 or args[2] in ("-", ".") else parse_time(args[2])
                segments = sp.set_bounds(segments, int(args[0]) - 1, start, end)
            elif cmd in ("t", "g", "l"):
                value = float(args[0])
                if cmd == "t":
                    params.threshold_db = value
                elif cmd == "g":
                    params.min_gap_s = value
                else:
                    params.min_track_s = value
                segments = sp.detect_segments(audio.envelope, audio.block_s, params)
            else:
                console.print("[red]Unknown command.[/red]")
        except (IndexError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")


# ----------------------------------------------------------------------------- side matching

def side_match_score(segments: list[sp.Segment], side: Side) -> float | None:
    """How badly detected tracks fit a side's expected track list, in seconds (lower = better).

    None when the side has no track lengths to compare against.
    """
    if not side.tracks or any(t.length_s is None for t in side.tracks):
        return None
    n = min(len(segments), len(side.tracks))
    score = sum(abs(segments[i].length - side.tracks[i].length_s) for i in range(n))
    score += 60.0 * abs(len(segments) - len(side.tracks))
    return score


def better_matching_side(segments: list[sp.Segment], current: int, candidates: list[int], planned: list[Side],
                         margin_s: float = 10.0, max_avg_error_s: float = 20.0) -> tuple[int, float, float] | None:
    """Return (index, its score, current score) if another candidate side fits clearly better.

    The alternative must actually fit: same track count and an average length error of at
    most `max_avg_error_s`, so a recording that matches nothing does not trigger a suggestion.
    """
    cur = side_match_score(segments, planned[current])
    if cur is None:
        return None
    best: tuple[float, int] | None = None
    for i in candidates:
        if i == current or len(planned[i].tracks) != len(segments):
            continue
        sc = side_match_score(segments, planned[i])
        if sc is None or sc / max(1, len(segments)) > max_avg_error_s:
            continue
        if best is None or sc < best[0]:
            best = (sc, i)
    if best and best[0] + margin_s < cur:
        return best[1], best[0], cur
    return None


def side_title(side: Side, multi_disc: bool) -> str:
    return f"side {side.label}" + (f" (disc {side.disc})" if multi_disc else "")


# ----------------------------------------------------------------------------- rip

@dataclass
class Recorded:
    disc_no: int
    side_label: str
    track: Track
    out_path: Path
    future: Future


@dataclass
class RipOptions:
    artist: str = ""
    album: str = ""
    files: list[Path] | None = None
    sides: list[str] | None = None      # side labels for --from-file inputs, in order
    no_auto_stop: bool = False


def _find_side(planned: list[Side], letter: str) -> int | None:
    matches = [i for i, s in enumerate(planned) if s.label.upper() == letter.upper()]
    return matches[0] if matches else None


def run_rip(cfg: Config, device: InputDevice | None, opts: RipOptions, console: Console) -> list[Path]:
    from_files = bool(opts.files)
    if not from_files and device is None:
        raise ValueError("an input device is required unless --from-file is used")

    if device is not None and not from_files:
        console.print(f"[bold]Input:[/bold] {device.describe()}  →  {cfg.audio.sample_rate} Hz / {cfg.audio.bit_depth}-bit FLAC")
        if device.sample_rate and device.sample_rate != cfg.audio.sample_rate:
            console.print(f"[yellow]Note: device runs at {device.sample_rate} Hz; audio will be resampled to {cfg.audio.sample_rate} Hz.[/yellow]")

    release = choose_release(console, cfg, opts.artist, opts.album)
    out_dir = album_dir(cfg, release)
    raw_dir = cfg.paths.raw_dir / sanitize(f"{release.artist} - {release.title}")
    console.print(f"[bold]Output:[/bold] {out_dir}")
    if not from_files:
        console.print(f"[bold]Raw sides:[/bold] {raw_dir}")

    planned = release.sides()
    fixed_plan = bool(planned) and all(s.known for s in planned)
    multi = len(release.discs) > 1
    queues: dict[int, list[Track]] = {}
    if planned and not fixed_plan:
        for d in release.discs:
            queues[d.number] = list(d.tracks)

    executor = ThreadPoolExecutor(max_workers=2)
    recorded: list[Recorded] = []
    done: set[int] = set()            # indexes into `planned` already recorded
    forced: int | None = None         # side chosen explicitly at the cue prompt
    side_index = 0                    # count of sides recorded (labels for unplanned sides)
    files = list(opts.files or [])
    file_sides = list(opts.sides or [])

    try:
        while True:
            # ---- decide what this side is
            idx: int | None = None
            if fixed_plan:
                remaining = [i for i in range(len(planned)) if i not in done]
                if forced is not None:
                    idx, forced = forced, None
                elif from_files and side_index < len(file_sides):
                    idx = _find_side(planned, file_sides[side_index])
                    if idx is None:
                        console.print(f"[red]No side '{file_sides[side_index]}' in this release.[/red]")
                        raise Abort()
                elif remaining:
                    idx = remaining[0]
                if idx is not None:
                    disc_no, label, expected = planned[idx].disc, planned[idx].label, planned[idx].tracks
                else:
                    if from_files and side_index < len(files):
                        console.print("[yellow]More files than sides in the release; extra tracks get placeholder titles.[/yellow]")
                    elif from_files or not ask(console, "All sides recorded. Record an extra side? [y/N]", "n").lower().startswith("y"):
                        break
                    disc_no = release.discs[-1].number
                    label = chr(ord("A") + side_index)
                    expected: list[Track] | None = []
            elif queues:
                remaining_discs = [n for n, q in queues.items() if q]
                if not remaining_discs:
                    if from_files and side_index < len(files):
                        console.print("[yellow]More files than tracks left in the release; extra tracks get placeholder titles.[/yellow]")
                        disc_no = release.discs[-1].number
                    else:
                        break
                else:
                    disc_no = remaining_discs[0]
                label = file_sides[side_index] if side_index < len(file_sides) else chr(ord("A") + side_index)
                expected = queues.get(disc_no, [])
            else:
                disc_no = min(side_index // 2 + 1, len(release.discs))
                label = file_sides[side_index] if side_index < len(file_sides) else chr(ord("A") + side_index)
                expected = None

            where = f"side {label}" + (f" (disc {disc_no})" if multi else "")

            # ---- capture
            if from_files:
                if side_index >= len(files):
                    break
                path = files[side_index]
                console.rule(f"{where}: {path.name}")
                with console.status(f"Analysing {path.name}..."):
                    audio = analyze_file(path, cfg, console)
            else:
                console.rule(f"Side {label}" + (f" · disc {disc_no}" if multi else ""))
                if expected:
                    console.print("Expected: " + ", ".join(f"[cyan]{t.title}[/cyan]" for t in expected))
                elif expected is None:
                    console.print("[dim]Track titles will be asked for after recording.[/dim]")
                hint = "type a side letter if you are playing a different side, " if fixed_plan or len(planned) == 0 else ""
                action = ask(console, f"Cue {where}, then press Enter to record ({hint}[d] done, [q] quit)", "")
                if action.lower() == "q":
                    raise Abort()
                if action.lower() == "d":
                    break
                if action and action.isalpha() and len(action) <= 2:
                    letter = action.upper()
                    if fixed_plan:
                        target = _find_side(planned, letter)
                        if target is None:
                            console.print(f"[red]This release has no side {letter}.[/red]")
                            continue
                        if target in done and not ask(console, f"Side {letter} was already recorded. Record it again? [y/N]", "n").lower().startswith("y"):
                            continue
                        forced = target
                        continue           # back to the top with the chosen side
                    label = letter
                    where = f"side {label}" + (f" (disc {disc_no})" if multi else "")
                elif action:
                    console.print("[red]Not understood.[/red]")
                    continue
                audio = record_side(device, cfg, raw_dir / f"side-{label}.wav", label, console, auto_stop=not opts.no_auto_stop)
                how = {"auto": "stopped after trailing silence", "user": "stopped by you", "eof": "input ended"}[audio.stopped_by]
                console.print(f"Recorded {fmt_time(audio.duration).strip()} ({how}).")
                if audio.extra.get("overflows"):
                    console.print(f"[red]Warning: {audio.extra['overflows']} input overflow(s) – some audio may be missing.[/red]")

            # ---- split
            segments = review_segments(console, cfg, label, audio, expected, allow_redo=not from_files)
            if segments is None:
                console.print("[yellow]Re-recording this side.[/yellow]")
                forced = idx
                continue

            # ---- does this look like a different side than planned?
            if fixed_plan and idx is not None:
                others = [i for i in range(len(planned)) if i not in done and i != idx]
                hit = better_matching_side(segments, idx, others, planned)
                if hit:
                    j, sc, cur = hit
                    console.print(f"[yellow]These tracks fit {side_title(planned[j], multi)} better "
                                  f"(off by {sc:.0f}s) than {side_title(planned[idx], multi)} (off by {cur:.0f}s).[/yellow]")
                    if ask(console, f"Treat this recording as side {planned[j].label}? [Y/n]", "y").lower().startswith("y"):
                        idx = j
                        disc_no, label, expected = planned[j].disc, planned[j].label, planned[j].tracks
                        if not from_files:
                            new_path = raw_dir / f"side-{label}.wav"
                            audio.path.replace(new_path)
                            audio.path = new_path

            # ---- assign tracks
            disc = next(d for d in release.discs if d.number == disc_no)
            side_tracks: list[Track] = []
            if fixed_plan:
                for i, seg in enumerate(segments):
                    if expected is not None and i < len(expected):
                        side_tracks.append(expected[i])
                    else:
                        t = Track(title="", position=f"{label}{i + 1}", number=len(disc.tracks) + 1)
                        t.title = ask(console, f"Title for extra track {i + 1} ({fmt_time(seg.length).strip()})", f"Track {t.number}")
                        disc.tracks.append(t)
                        side_tracks.append(t)
                if expected and len(segments) < len(expected):
                    missing = ", ".join(t.title for t in expected[len(segments):])
                    console.print(f"[yellow]Not recorded on this side: {missing}[/yellow]")
            elif queues:
                queue = queues[disc_no]
                for i, seg in enumerate(segments):
                    if queue:
                        side_tracks.append(queue.pop(0))
                    else:
                        t = Track(title="", position=f"{label}{i + 1}", number=len(disc.tracks) + 1)
                        t.title = ask(console, f"Title for extra track {i + 1} ({fmt_time(seg.length).strip()})", f"Track {t.number}")
                        disc.tracks.append(t)
                        side_tracks.append(t)
            else:
                console.print("[bold]Track titles[/bold] (Enter keeps the default)")
                for i, seg in enumerate(segments):
                    n = len(disc.tracks) + 1
                    title = ask(console, f"  {label}{i + 1}  {fmt_time(seg.length).strip()}  title", f"Track {n}")
                    t = Track(title=title, position=f"{label}{i + 1}", number=n)
                    disc.tracks.append(t)
                    side_tracks.append(t)

            # ---- schedule encodes
            for seg, track in zip(segments, side_tracks):
                out_path = out_dir / track_filename(cfg, release, track, disc_no, label)
                job = TrackJob(audio.path, seg.start, seg.end, out_path, build_tags(release, track, disc_no, label),
                               pair=audio.pair if audio.channels > 2 else None)
                fut = executor.submit(encode_track, job, cfg)
                recorded.append(Recorded(disc_no, label, track, out_path, fut))
            console.print(f"[green]Queued {len(side_tracks)} track(s) for encoding.[/green]")
            if idx is not None:
                done.add(idx)
            side_index += 1

            if fixed_plan and not from_files:
                left = [planned[i] for i in range(len(planned)) if i not in done]
                if left:
                    console.print("Still to record: " + ", ".join(side_title(s, multi) for s in left))
            if not fixed_plan and not queues and not from_files:
                nxt = ask(console, "Record another side? [Y/n]", "y")
                if not nxt.lower().startswith("y"):
                    break
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        executor.shutdown(wait=True, cancel_futures=True)
        raise Abort()

    # ---- finish
    if not recorded:
        executor.shutdown()
        console.print("[yellow]Nothing recorded.[/yellow]")
        return []
    with console.status("Waiting for encoding to finish..."):
        executor.shutdown(wait=True)
    done_files: list[Path] = []
    failures = 0
    for r in recorded:
        try:
            r.future.result()
        except Exception as exc:
            failures += 1
            console.print(f"[red]{r.out_path.name}: {exc}[/red]")
            continue
        # Final tag pass with the complete release (track totals, titles entered late, cover).
        tag_flac(r.out_path, build_tags(release, r.track, r.disc_no, r.side_label), release.cover, release.cover_mime)
        done_files.append(r.out_path)

    table = Table(title=f"{release.artist} – {release.title}")
    table.add_column("Disc/Track", justify="right")
    table.add_column("Title")
    table.add_column("File")
    for r in sorted(recorded, key=lambda r: (r.disc_no, r.track.number)):
        if r.out_path in done_files:
            table.add_row(f"{r.disc_no}-{r.track.number:02d}", r.track.title, str(r.out_path.relative_to(cfg.paths.music_dir)))
    console.print(table)
    if failures:
        console.print(f"[red]{failures} track(s) failed to encode. Raw recordings kept in {raw_dir}[/red]")
    elif not from_files and not cfg.paths.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        console.print("[dim]Raw side recordings deleted (paths.keep_raw = false).[/dim]")
    return done_files
