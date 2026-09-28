"""Release metadata from MusicBrainz and Discogs, normalised to a common model."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import requests

from . import __version__
from .config import MetadataConfig

MB_ROOT = "https://musicbrainz.org/ws/2"
CAA_ROOT = "https://coverartarchive.org"
DISCOGS_ROOT = "https://api.discogs.com"


@dataclass
class Track:
    title: str
    position: str = ""            # "A1", "B3", "1", "2-4" ...
    artist: str = ""
    number: int = 0               # 1-based within the disc
    length_s: float | None = None
    mb_recording_id: str = ""
    mb_track_id: str = ""

    @property
    def side(self) -> str | None:
        m = re.match(r"^\s*([A-Za-z]+)\s*\d*\s*$", self.position)
        return m.group(1).upper() if m else None


@dataclass
class Disc:
    number: int
    tracks: list[Track] = field(default_factory=list)
    format: str = ""
    title: str = ""


@dataclass
class Side:
    disc: int
    label: str                    # "A", "B", ...
    tracks: list[Track]
    known: bool = True            # False when the source did not say which tracks are on which side


@dataclass
class Candidate:
    source: str                   # "musicbrainz" | "discogs"
    id: str
    title: str
    artist: str
    year: str = ""
    country: str = ""
    label: str = ""
    catno: str = ""
    format: str = ""
    track_count: int | None = None
    disc_count: int = 1


@dataclass
class Release:
    artist: str
    title: str
    year: str = ""
    label: str = ""
    catno: str = ""
    country: str = ""
    discs: list[Disc] = field(default_factory=list)
    mb_release_id: str = ""
    discogs_id: str = ""
    cover: bytes | None = None
    cover_mime: str = "image/jpeg"
    source: str = "manual"

    @property
    def all_tracks(self) -> list[Track]:
        return [t for d in self.discs for t in d.tracks]

    def sides(self) -> list[Side]:
        """Group tracks per disc into sides using their positions (A1, B2...)."""
        out: list[Side] = []
        for disc in self.discs:
            if not disc.tracks:
                continue
            letters = [t.side for t in disc.tracks]
            if all(letters):
                groups: dict[str, list[Track]] = {}
                for t, letter in zip(disc.tracks, letters):
                    groups.setdefault(letter, []).append(t)
                for letter, tracks in groups.items():
                    out.append(Side(disc.number, letter, tracks, known=True))
            else:
                out.append(Side(disc.number, "", list(disc.tracks), known=False))
        return out


class MetadataError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- helpers

def _user_agent(cfg: MetadataConfig) -> str:
    contact = cfg.musicbrainz_contact.strip() or "https://github.com/vinylrip"
    return f"vinylrip/{__version__} ( {contact} )"


def _lucene_escape(s: str) -> str:
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", s)


def _words(text: str) -> list[str]:
    """Alphanumeric words of a title, for loose matching."""
    return [w for w in re.split(r"[^\w']+", text) if w]


def parse_length(text: str) -> float | None:
    """'3:45' -> 225.0 ; '' -> None."""
    text = (text or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return total


class _RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self.last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last = time.monotonic()


# ----------------------------------------------------------------------------- MusicBrainz

class MusicBrainz:
    def __init__(self, cfg: MetadataConfig) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = _user_agent(cfg)
        self.session.headers["Accept"] = "application/json"
        self.limiter = _RateLimiter(1.1)

    def _get(self, path: str, **params) -> dict:
        self.limiter.wait()
        params["fmt"] = "json"
        resp = self.session.get(f"{MB_ROOT}/{path}", params=params, timeout=30)
        if resp.status_code == 503:
            time.sleep(2)
            resp = self.session.get(f"{MB_ROOT}/{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search(self, artist: str, album: str, limit: int = 12) -> list[Candidate]:
        terms = []
        if artist:
            terms.append(f'artist:"{_lucene_escape(artist)}"')
        if album:
            terms.append(f'release:"{_lucene_escape(album)}"')
        if not terms:
            return []
        base = " AND ".join(terms)
        # Looser variant: individual words, so "Abandonnée/Maléja" still finds "Abandonnée / Maleja".
        loose_terms = []
        if artist:
            loose_terms.append("artist:(" + " AND ".join(_lucene_escape(w) for w in _words(artist)) + ")")
        if album:
            loose_terms.append("release:(" + " AND ".join(_lucene_escape(w) for w in _words(album)) + ")")
        loose = " AND ".join(loose_terms)
        vinyl_formats = " OR ".join(f'format:"{f}"' for f in ('12\\" Vinyl', '10\\" Vinyl', '7\\" Vinyl', "Vinyl"))
        # Vinyl pressings first (exact, then loose), then anything else with the same name.
        queries = [f"{base} AND ({vinyl_formats})", f"{loose} AND ({vinyl_formats})", base, loose]
        releases: list[dict] = []
        seen: set[str] = set()
        for query in dict.fromkeys(queries):
            if len(releases) >= limit:
                break
            for rel in self._get("release", query=query, limit=limit).get("releases", []):
                if rel["id"] not in seen:
                    seen.add(rel["id"])
                    releases.append(rel)
        out = []
        for rel in releases:
            media = rel.get("media", []) or []
            formats = sorted({m.get("format", "") for m in media if m.get("format")})
            labels = rel.get("label-info", []) or []
            label = catno = ""
            for li in labels:
                if li.get("label", {}).get("name"):
                    label = li["label"]["name"]
                    catno = li.get("catalog-number", "") or ""
                    break
            out.append(
                Candidate(
                    source="musicbrainz",
                    id=rel["id"],
                    title=rel.get("title", ""),
                    artist=_credit(rel.get("artist-credit", [])),
                    year=(rel.get("date") or "")[:4],
                    country=rel.get("country", "") or "",
                    label=label,
                    catno=catno,
                    format=", ".join(formats),
                    track_count=rel.get("track-count"),
                    disc_count=len(media) or 1,
                )
            )
        vinyl = [c for c in out if "vinyl" in c.format.lower()]
        others = [c for c in out if c not in vinyl]
        return (vinyl + others)[:limit]

    def release(self, mbid: str) -> Release:
        data = self._get(f"release/{mbid}", inc="recordings+artist-credits+labels+media")
        labels: list[str] = []
        catnos: list[str] = []
        for li in data.get("label-info", []) or []:
            name = (li.get("label") or {}).get("name")
            if name and name not in labels:
                labels.append(name)
            cat = li.get("catalog-number") or ""
            if cat and cat not in catnos:
                catnos.append(cat)
        label = " / ".join(labels)
        catno = " / ".join(catnos)
        discs = []
        for medium in data.get("media", []) or []:
            disc = Disc(number=int(medium.get("position", len(discs) + 1)), format=medium.get("format", "") or "",
                        title=medium.get("title", "") or "")
            for i, t in enumerate(medium.get("tracks", []) or [], start=1):
                rec = t.get("recording", {}) or {}
                length_ms = t.get("length") or rec.get("length")
                disc.tracks.append(
                    Track(
                        title=t.get("title") or rec.get("title", ""),
                        position=str(t.get("number", "") or i),
                        artist=_credit(t.get("artist-credit") or rec.get("artist-credit") or []),
                        number=int(t.get("position", i)),
                        length_s=(length_ms / 1000.0) if length_ms else None,
                        mb_recording_id=rec.get("id", ""),
                        mb_track_id=t.get("id", ""),
                    )
                )
            discs.append(disc)
        return Release(
            artist=_credit(data.get("artist-credit", [])),
            title=data.get("title", ""),
            year=(data.get("date") or "")[:4],
            label=label,
            catno=catno,
            country=data.get("country", "") or "",
            discs=discs,
            mb_release_id=mbid,
            source="musicbrainz",
        )

    def cover(self, mbid: str) -> tuple[bytes, str] | None:
        try:
            resp = self.session.get(f"{CAA_ROOT}/release/{mbid}/front-500", timeout=30, allow_redirects=True)
        except requests.RequestException:
            return None
        if resp.status_code != 200 or not resp.content:
            return None
        return resp.content, resp.headers.get("Content-Type", "image/jpeg").split(";")[0]


def _credit(credit: list) -> str:
    out = ""
    for part in credit or []:
        if isinstance(part, dict):
            out += part.get("name") or part.get("artist", {}).get("name", "")
            out += part.get("joinphrase", "") or ""
        else:
            out += str(part)
    return out.strip()


# ----------------------------------------------------------------------------- Discogs

class Discogs:
    def __init__(self, cfg: MetadataConfig) -> None:
        self.token = cfg.discogs_token.strip()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = _user_agent(cfg)
        if self.token:
            self.session.headers["Authorization"] = f"Discogs token={self.token}"
        self.limiter = _RateLimiter(1.0)

    @property
    def available(self) -> bool:
        return True  # search works without a token (lower rate limit, no images)

    def _get(self, path: str, **params) -> dict:
        self.limiter.wait()
        resp = self.session.get(f"{DISCOGS_ROOT}/{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search(self, artist: str, album: str, limit: int = 12) -> list[Candidate]:
        if not self.available:
            return []
        params = {"type": "release", "format": "Vinyl", "per_page": limit}
        if artist:
            params["artist"] = artist
        if album:
            params["release_title"] = album
        data = self._get("database/search", **params)
        out = []
        for r in data.get("results", []):
            title = r.get("title", "")
            art, _, alb = title.partition(" - ")
            fmt = r.get("format", []) or []
            out.append(
                Candidate(
                    source="discogs",
                    id=str(r["id"]),
                    title=_clean_discogs_name(alb or title),
                    artist=_clean_discogs_name(art) if alb else "",
                    year=str(r.get("year", "") or ""),
                    country=r.get("country", "") or "",
                    label=(r.get("label") or [""])[0],
                    catno=r.get("catno", "") or "",
                    format=", ".join(fmt[:3]),
                    track_count=None,
                )
            )
        return out

    def release(self, discogs_id: str) -> Release:
        data = self._get(f"releases/{discogs_id}")
        artist = _discogs_credit(data.get("artists", []) or [])
        label = catno = ""
        for lab in data.get("labels", []) or []:
            label = lab.get("name", "")
            catno = lab.get("catno", "") or ""
            break

        flat: list[Track] = []
        for item in data.get("tracklist", []) or []:
            items = item.get("sub_tracks") if item.get("type_") == "index" else [item]
            for t in items or []:
                if t.get("type_", "track") != "track":
                    continue
                t_art = _discogs_credit(t.get("artists", []) or [])
                flat.append(
                    Track(
                        title=t.get("title", ""),
                        position=t.get("position", "") or "",
                        artist=t_art,
                        length_s=parse_length(t.get("duration", "")),
                    )
                )
        discs = _discs_from_positions(flat)

        cover = None
        mime = "image/jpeg"
        images = data.get("images", []) or []
        primary = next((i for i in images if i.get("type") == "primary"), images[0] if images else None)
        if primary and primary.get("uri") and self.token:  # image downloads need a token
            try:
                self.limiter.wait()
                resp = self.session.get(primary["uri"], timeout=30)
                if resp.status_code == 200:
                    cover = resp.content
                    mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            except requests.RequestException:
                pass

        return Release(
            artist=artist,
            title=data.get("title", ""),
            year=str(data.get("year", "") or ""),
            label=label,
            catno=catno,
            country=data.get("country", "") or "",
            discs=discs,
            discogs_id=str(discogs_id),
            cover=cover,
            cover_mime=mime,
            source="discogs",
        )


def _discogs_credit(artists: list) -> str:
    out = ""
    for a in artists or []:
        out += _clean_discogs_name(a.get("anv") or a.get("name", ""))
        join = (a.get("join") or "").strip()
        if join:
            out += f"{join} " if join in (",", ";") else f" {join} "
    return out.strip()


def _clean_discogs_name(name: str) -> str:
    """Strip Discogs artefacts: 'Name (2)' disambiguation and 'Name*' name-variation markers."""
    name = re.sub(r"\s\(\d+\)(?=\s|,|$)", "", name)
    return name.replace("*", "").strip()


def _discs_from_positions(tracks: list[Track]) -> list[Disc]:
    """Assign tracks to discs from Discogs-style positions (A1..D4 or 1-1, 2-3)."""
    discs: dict[int, Disc] = {}
    letter_order: list[str] = []
    for t in tracks:
        disc_no = 1
        m = re.match(r"^\s*(\d+)[-.](\d+)\s*$", t.position)
        if m:
            disc_no = int(m.group(1))
        elif t.side:
            if t.side not in letter_order:
                letter_order.append(t.side)
            disc_no = letter_order.index(t.side) // 2 + 1
        disc = discs.setdefault(disc_no, Disc(number=disc_no))
        t.number = len(disc.tracks) + 1
        disc.tracks.append(t)
    return [discs[k] for k in sorted(discs)]


# ----------------------------------------------------------------------------- top level

class MetadataLookup:
    def __init__(self, cfg: MetadataConfig) -> None:
        self.cfg = cfg
        self.mb = MusicBrainz(cfg)
        self.discogs = Discogs(cfg)

    def search(self, artist: str, album: str) -> tuple[list[Candidate], list[str]]:
        warnings: list[str] = []
        results: list[Candidate] = []
        try:
            results += self.mb.search(artist, album)
        except requests.RequestException as exc:
            warnings.append(f"MusicBrainz search failed: {exc}")
        if self.discogs.available:
            try:
                results += self.discogs.search(artist, album)
            except requests.RequestException as exc:
                warnings.append(f"Discogs search failed: {exc}")
        return results, warnings

    def fetch(self, cand: Candidate) -> Release:
        if cand.source == "musicbrainz":
            rel = self.mb.release(cand.id)
            if self.cfg.embed_cover:
                cover = self.mb.cover(cand.id)
                if cover:
                    rel.cover, rel.cover_mime = cover
            return rel
        rel = self.discogs.release(cand.id)
        if not self.cfg.embed_cover:
            rel.cover = None
        return rel


def manual_release(artist: str, title: str, year: str = "", discs: int = 1) -> Release:
    return Release(artist=artist, title=title, year=year, discs=[Disc(number=i + 1) for i in range(max(1, discs))], source="manual")
