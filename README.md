# vinylrip

A command-line vinyl ripper for macOS and Linux. It records each side of a record
from your audio interface, splits the side into tracks by detecting the silence
between them, trims needle-drop and needle-lift pops, looks the release up on
MusicBrainz and Discogs and saves tagged FLAC files with cover art.

No GUI: everything happens in the terminal, including reviewing and adjusting the
detected track boundaries before anything is encoded.

## Requirements

* Python 3.11+
* `ffmpeg` and `ffprobe` on your PATH (`brew install ffmpeg` / `apt install ffmpeg`),
  used for cutting and FLAC encoding
* Capture uses PortAudio through the `sounddevice` package. macOS wheels bundle
  PortAudio; on Linux install it first (`apt install libportaudio2`). If PortAudio is
  unavailable, set `backend = "ffmpeg"` to capture through ffmpeg instead. (On macOS,
  ffmpeg's avfoundation input was measured dropping ~15% of samples, so prefer PortAudio.)

## Install

```sh
cd vinyl-ripper
python3 -m venv .venv && .venv/bin/pip install -e .
# then either use .venv/bin/vinylrip directly, or put it on your PATH:
ln -s "$PWD/.venv/bin/vinylrip" /usr/local/bin/vinylrip
```

`pipx install .` also works.

## Quick start

```sh
vinylrip config init          # writes ~/.config/vinylrip/config.toml with comments
vinylrip devices              # shows inputs; * marks the system default
vinylrip levels               # live meter: set your gain, note the noise floor
vinylrip rip                  # rip a record
```

A rip goes like this:

1. The input device, sample rate and bit depth are shown.
2. You enter artist and album. Matching releases from MusicBrainz (vinyl pressings
   first) and Discogs are listed; pick one, search again, or enter the details by hand.
   The chosen release's sides and track list are printed for confirmation.
3. For each side: cue the record, press Enter, then drop the needle when the meter says
   `capturing – drop the needle now`. The meter shows level, clipping, and how many
   tracks have been detected so far. The side stops by itself after the run-out
   silence (with a grace period so a quiet passage can be continued with `c`), or
   press Enter to stop it.
4. The detected tracks are listed next to the expected track list with lengths, so
   you can see at a glance whether the split matches. Adjust if needed:

   | command            | effect                                        |
   |--------------------|-----------------------------------------------|
   | Enter / `ok`       | accept                                        |
   | `m 2`              | merge track 2 with track 3                    |
   | `s 2 3:41`         | split track 2 at 3:41 (time within the side)  |
   | `d 4`              | delete track 4                                |
   | `a 0:12 4:05`      | add a track from 0:12 to 4:05                 |
   | `e 1 0:10 -`       | set track 1 start (use `-` to keep a value)   |
   | `t -40`            | re-detect with a -40 dBFS silence threshold   |
   | `g 2.0`            | re-detect with a 2 s minimum gap              |
   | `l 8`              | re-detect with an 8 s minimum track length    |
   | `r`                | re-record this side                           |

5. Flip the record (or load the next disc) and repeat. Tracks are encoded in the
   background while you do that. Multi-disc releases are handled by their side
   letters (A/B on disc 1, C/D on disc 2, ...). Sides can be played in any order:
   type the letter of the side you are actually playing at the cue prompt, and if the
   detected track lengths clearly fit a different remaining side better than the
   planned one, the tool says so and offers to file the recording under that side.
6. Files land in `music_dir/<Artist>/<Album (Year)>/NN - Title.flac`, tagged with
   title, artist, album, track/disc numbers, date, label, catalogue number,
   MusicBrainz/Discogs IDs and the front cover.

If a release cannot be found (or you pick manual entry), you are asked for artist,
album, year and disc count, and for track titles after each side.

## Other commands

```sh
vinylrip rip --from-file sideA.wav sideB.wav   # split & tag existing recordings
vinylrip rip --from-file c.wav a.wav --sides C,A # ...telling it which side each file is
vinylrip detect sideA.wav                      # dry run: show detected tracks
vinylrip rip -a "Fleetwood Mac" -r Rumours     # skip the search prompts
vinylrip rip -d "Xone" --rate 96000 --bits 24  # one-off overrides
vinylrip config show                           # effective settings
```

## Configuration

`vinylrip config init` writes a fully commented file. The main knobs:

* `[paths]` – `music_dir`, `raw_dir` (untouched 32-bit WAV of each side, kept by
  default), folder and file name templates.
* `[audio]` – `device` (name, index or `default`), `backend`, `sample_rate`,
  `bit_depth` (16/24), `input_channels` (`"auto"` records every channel of a
  multi-channel interface and uses the loudest adjacent pair, e.g. a DJ mixer that
  sends 4 channels over USB; or set the pair explicitly, `[2, 3]`).
* `[detection]` – silence threshold (`"auto"` derives it from the recording's own
  noise floor; or a fixed dBFS value), minimum gap between tracks, minimum track
  length, pop trimming, padding, fade length, and the end-of-side silence/grace times.
* `[metadata]` – `musicbrainz_contact` (MusicBrainz asks for one in the User-Agent),
  `discogs_token` (optional; raises the Discogs rate limit and enables Discogs cover art;
  get one at https://www.discogs.com/settings/developers), `embed_cover`.

Any option can also be pointed at a different file with `--config`.

## How splitting works

While recording, an RMS level is computed every 50 ms. Track detection works on that
envelope:

1. blocks above the silence threshold form "bursts"; bursts shorter than
   `min_sound_seconds` (clicks, lock-groove ticks) are ignored;
2. bursts closer together than `min_gap_seconds` are merged into one track;
3. a short burst at the start or end of a track that is separated from the music by a
   small gap (a needle drop or lift) is trimmed off;
4. anything shorter than `min_track_seconds` is discarded;
5. `pad_seconds` of the surrounding groove noise is kept on each side of a track and a
   short fade is applied at the cut.

If a record has quiet passages that get split, merge them with `m`, raise the gap with
`g`, or lower the threshold with `t`. `vinylrip levels` shows the noise floor of your
setup so you can pick a fixed threshold if `auto` does not suit it.

## Tests

```sh
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

`tests/make_side.py` generates a synthetic side (tones, gaps, pops) if you want to try
the pipeline without a turntable: `python tests/make_side.py side.wav 20 12 16`.
