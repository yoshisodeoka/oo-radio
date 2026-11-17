#!/usr/bin/env python3
"""Build a 10-minute sketch mix from the best collected takes.

Uses triage.py to find where each take actually has signal, then places
excerpts on a fixed five-part timeline:

  0:00  Carrier   the Buzzer fades in, an octave-down copy under it, WWV ticks
  1:30  Markers   the Pip and the Squeaky Wheel as rhythm, morse fragments
  3:30  Voices    numbers voices over a low Buzzer drone a fifth apart
  6:30  Polytone  XPA2 / XPB tone chains, a fifth-down echo, the Pip under it
  8:30  Time      CHU and WWV talking clocks, the Buzzer fades out

Commands:
  python sketch.py                 build recordings/sketch/sketch-<time>.wav and .mp3
  python sketch.py --seed 3        same timeline, different excerpts
  python sketch.py --minutes 5     squeeze the timeline to another length

Writes a cue sheet next to the audio listing every excerpt (take, offset,
station, frequency, receiver) for notes and credits. Needs numpy and ffmpeg.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np

import triage

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "recordings")
OUT_SR = 48000
CACHE = os.path.join(REC, ".sketch-windows.json")
XFADE = 2.0  # seconds of crossfade when one part needs several pieces

# Station pools: the first station in a pool with usable signal is used, and
# repeated requests rotate through the pool.
POOLS = {
    "buzz": ["uvb76"],
    "pip": ["pip"],
    "squeak": ["squeaky"],
    "morse": ["m12", "m14"],
    "voice": ["e11", "v13", "s06", "s11a", "e07", "v32"],
    "poly_a": ["xpa2", "xpa", "xpb"],
    "poly_b": ["xpb", "xpa2", "xpa"],
    "time_us": ["wwv", "chu"],
    "time_ca": ["chu", "wwv"],
}

# (pool, start s, length s, gain dB, pan -1..1, pitch ratio, fade in s, fade out s, filters)
# Times are for a 600 s sketch; --minutes scales them.
SCORE = [
    # Carrier
    ("buzz",     0, 160,  -6,  0.0, 1.0,  25, 20, "lowpass=f=2400"),
    ("buzz",    20, 140, -13, -0.5, 0.5,  30, 20, "lowpass=f=1200"),
    ("time_us", 40,  60, -15,  0.6, 1.0,  10, 10, "highpass=f=300"),
    # Markers
    ("pip",     90, 120,  -8, -0.4, 1.0,  15, 15, "lowpass=f=3000"),
    ("squeak", 120, 100, -11,  0.5, 1.0,  15, 15, "lowpass=f=3000"),
    ("morse",  140,  25, -13,  0.7, 1.0,   3,  5, "bandpass=f=800:w=600"),
    ("morse",  180,  25, -13, -0.7, 1.0,   3,  5, "bandpass=f=800:w=600"),
    ("buzz",   150, 260, -12,  0.0, 0.5,  30, 30, "lowpass=f=900"),
    # Voices, over the drone a fifth apart
    ("buzz",   210, 190, -15,  0.4, 0.75, 30, 30, "lowpass=f=900"),
    ("voice",  220,  35,  -5, -0.3, 1.0,   2,  4, "highpass=f=250,lowpass=f=3200"),
    ("voice",  262,  35,  -5,  0.3, 1.0,   2,  4, "highpass=f=250,lowpass=f=3200"),
    ("voice",  304,  35,  -5,  0.0, 1.0,   2,  4, "highpass=f=250,lowpass=f=3200"),
    ("voice",  346,  35,  -5, -0.4, 1.0,   2,  4, "highpass=f=250,lowpass=f=3200"),
    ("morse",  330,  20, -15,  0.6, 1.0,   3,  4, "bandpass=f=800:w=600"),
    # Polytone
    ("poly_a", 390,  70,  -6, -0.3, 1.0,   8, 10, "lowpass=f=3500"),
    ("poly_b", 430,  80,  -6,  0.3, 1.0,   8, 10, "lowpass=f=3500"),
    ("poly_a", 405,  85, -15,  0.0, 0.667, 15, 15, "lowpass=f=2000"),
    ("pip",    420,  90, -13, -0.6, 1.0,  15, 15, "lowpass=f=3000"),
    ("squeak", 470,  40, -15,  0.6, 1.0,  10, 10, "lowpass=f=3000"),
    # Time
    ("time_ca", 505, 65, -10, -0.4, 1.0,  10, 10, "highpass=f=200,lowpass=f=3500"),
    ("time_us", 520, 75,  -8,  0.4, 1.0,  10, 15, "highpass=f=200,lowpass=f=3500"),
    ("buzz",   495, 105,  -9,  0.0, 1.0,  20, 35, "lowpass=f=2400"),
]

MASTER = ("highpass=f=40,"
          "aecho=0.8:0.6:70|150|310|560:0.35|0.28|0.2|0.14,"
          "alimiter=limit=0.9,loudnorm=I=-16:TP=-1.5:LRA=11")


# ---------------------------------------------------------------- takes

def all_takes():
    files = sorted(glob.glob(os.path.join(REC, "????-??-??", "*.flac")) +
                   glob.glob(os.path.join(REC, "????-??-??", "*.wav")))
    return [f for f in files if triage.take_info(f)]


def triage_rows():
    path = os.path.join(REC, "triage.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {os.path.normcase(os.path.normpath(r["file"])): r for r in csv.DictReader(f)}


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def windows_for(path, cache):
    """Signal flags per 10 s window, from triage's detector (cached)."""
    key = os.path.relpath(path, HERE)
    stamp = os.path.getmtime(path)
    hit = cache.get(key)
    if hit and hit["mtime"] == stamp:
        return hit["signal"]
    feats = triage.window_features(triage.decode(path))
    sig = [bool(triage.is_signal(f)) for f in feats]
    cache[key] = {"mtime": stamp, "signal": sig}
    return sig


class Library:
    """Finds unused stretches of signal for a station, best takes first."""

    def __init__(self, rng):
        self.rng = rng
        self.cache = load_cache()
        rows = triage_rows()
        self.by_station = {}
        for path in all_takes():
            info = triage.take_info(path)
            row = rows.get(os.path.normcase(os.path.normpath(info["file"])))
            if row and row["verdict"] == "static":
                continue  # already known to be static: skip decoding it
            share = float(row["signal_share"]) if row else 0.5
            self.by_station.setdefault(info["station"], []).append([share, path, info])
        for takes in self.by_station.values():
            takes.sort(key=lambda t: -t[0])
        self.used = {}      # path -> list of (start, end) seconds already placed
        self.turn = {}      # pool -> rotation counter

    def runs(self, path):
        sig = windows_for(path, self.cache)
        out, start = [], None
        for i, s in enumerate(sig + [False]):
            if s and start is None:
                start = i
            elif not s and start is not None:
                out.append((start * triage.WIN, i * triage.WIN))
                start = None
        return out

    def free(self, path, a, b):
        return all(b <= u0 or a >= u1 for u0, u1 in self.used.get(path, []))

    def find(self, station, need, reuse=False):
        """The longest unused stretch of signal (up to need seconds) for this station."""
        best = None
        for share, path, info in self.by_station.get(station, [])[:8]:
            for a, b in self.runs(path):
                length = min(need, b - a)
                if length < min(need, 20):
                    continue
                slack = (b - a) - length
                for _ in range(6):
                    off = a + (self.rng.uniform(0, slack) if slack > 0 else 0)
                    if reuse or self.free(path, off, off + length):
                        # prefer longer pieces; the seed breaks near-ties
                        key = length + self.rng.uniform(0, 10)
                        if best is None or key > best[0]:
                            best = (key, path, info, off, length)
                        break
        if best:
            _, path, info, off, length = best
            self.used.setdefault(path, []).append((off, off + length))
            return path, info, off, length
        return None

    def fill(self, pool, need):
        """Pieces from one pool adding up to need seconds (joined with crossfades later).
        Unused material first; already-placed material only when nothing else is left."""
        stations = POOLS[pool]
        k = self.turn.get(pool, 0)
        self.turn[pool] = k + 1
        order = stations[k % len(stations):] + stations[:k % len(stations)]
        for reuse in (False, True):
            for st in order:
                pieces, have = [], 0
                while have < need and len(pieces) < 6:
                    hit = self.find(st, need - have + (XFADE if pieces else 0), reuse)
                    if not hit:
                        break
                    pieces.append(hit)
                    have += hit[3] - (XFADE if len(pieces) > 1 else 0)
                if pieces and have >= min(need, 20):
                    return pieces
        return []

    def save(self):
        with open(CACHE, "w") as f:
            json.dump(self.cache, f)


# ---------------------------------------------------------------- audio

def render_excerpt(path, offset, length, pitch, filters):
    """Cut, filter and (tape-style) pitch an excerpt; returns mono float at OUT_SR."""
    src_len = length * pitch  # slower playback needs less source for the same output time
    chain = [f"aresample={triage.SR}", "aformat=channel_layouts=mono"]
    if filters:
        chain.append(filters)
    chain += [f"asetrate={triage.SR * pitch:.0f}", f"aresample={OUT_SR}"]
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{offset:.2f}", "-t", f"{src_len:.2f}",
         "-i", path, "-af", ",".join(chain), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, check=True).stdout
    a = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    peak = np.percentile(np.abs(a), 99.5) if len(a) else 0
    return a / peak * 0.5 if peak > 0 else a


def crossfade(a, b, seconds):
    n = min(int(seconds * OUT_SR), len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    ramp = np.linspace(0, np.pi / 2, n)
    mid = a[-n:] * np.cos(ramp) + b[:n] * np.sin(ramp)
    return np.concatenate([a[:-n], mid, b[n:]])


def envelope(n, fade_in, fade_out):
    env = np.ones(n)
    fi, fo = min(n, int(fade_in * OUT_SR)), min(n, int(fade_out * OUT_SR))
    if fi:
        env[:fi] = np.sin(np.linspace(0, np.pi / 2, fi)) ** 2
    if fo:
        env[n - fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
    return env


def build(args):
    rng = random.Random(args.seed)
    lib = Library(rng)
    if not lib.by_station:
        sys.exit("no takes found in recordings/YYYY-MM-DD/")
    scale = args.minutes * 60 / 600
    total = int(args.minutes * 60 * OUT_SR)
    mix = np.zeros((2, total))
    cues = []

    for i, (pool, start, length, gain, pan, pitch, fi, fo, filters) in enumerate(SCORE, 1):
        start, length = start * scale, max(10, length * scale)
        need = int(np.ceil(length * pitch))
        pieces = lib.fill(pool, need)
        if not pieces:
            print(f"  {i:2d} {pool:<8} nothing usable, left out", file=sys.stderr)
            continue
        a = None
        for path, info, off, got in pieces:
            print(f"  {i:2d} {pool:<8} {info['station']:<8} {os.path.basename(path)} "
                  f"@ {int(off) // 60}:{int(off) % 60:02d} for {got:.0f}s", file=sys.stderr, flush=True)
            b = render_excerpt(path, off, got / pitch, pitch, filters)
            a = b if a is None else crossfade(a, b, XFADE)
            cues.append({"at": f"{int(start) // 60}:{int(start) % 60:02d}", "part": pool,
                         "station": info["station"], "khz": info["khz"], "receiver": info["receiver"],
                         "take": info["file"], "take_offset": f"{int(off) // 60}:{int(off) % 60:02d}",
                         "take_seconds": round(got, 1), "pitch": pitch, "gain_db": gain, "pan": pan})
        a = a[: int(length * OUT_SR)]
        s = int(start * OUT_SR)
        a = a[: max(0, total - s)]
        a *= envelope(len(a), fi * scale, fo * scale) * 10 ** (gain / 20)
        th = (pan + 1) * np.pi / 4
        mix[0, s:s + len(a)] += a * np.cos(th)
        mix[1, s:s + len(a)] += a * np.sin(th)
    lib.save()

    out_dir = os.path.join(REC, "sketch")
    os.makedirs(out_dir, exist_ok=True)
    name = f"sketch-{dt.datetime.now().strftime('%Y%m%d-%H%M')}-seed{args.seed}"
    wav = os.path.join(out_dir, name + ".wav")
    peak = np.abs(mix).max() or 1
    pcm = (mix.T / peak * 0.7).astype(np.float32).tobytes()
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(OUT_SR),
                    "-ac", "2", "-i", "-", "-af", MASTER, "-ar", str(OUT_SR), "-c:a", "pcm_s24le", wav],
                   input=pcm, check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", wav, "-b:a", "320k",
                    os.path.join(out_dir, name + ".mp3")], check=True)
    with open(os.path.join(out_dir, name + "-cues.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cues[0].keys()) if cues else ["at"])
        w.writeheader()
        w.writerows(cues)
    print(f"\n{len({c['at'] + c['part'] for c in cues})} of {len(SCORE)} parts placed, from {len(cues)} excerpts")
    print(f"Wrote {os.path.relpath(wav, HERE)} (+ .mp3 and -cues.csv)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--minutes", type=float, default=10)
    build(p.parse_args())


if __name__ == "__main__":
    main()
