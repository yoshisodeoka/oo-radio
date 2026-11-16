#!/usr/bin/env python3
"""Sort the collected recordings into worth-a-listen and static.

Every take is cut into 10-second windows. A window counts as signal when, in
enough of its 85 ms slices, some frequency stands clear of that take's own
background (tones, buzz harmonics, morse, voice), or when such slices recur in
a steady rhythm (the Buzzer, the Pip). Plain band noise, fading, and constant
interference lines don't count.

Commands:
  python triage.py                 score new takes, then print the report
  python triage.py --rescore       score every take again (after changing thresholds)
  python triage.py --dump FILE...  print the raw per-window numbers (for tuning)

Writes recordings/triage.csv (one row per take) and recordings/listen.txt
(the best takes per station, with the time of the strongest minute so you
can jump straight to it). Needs numpy and ffmpeg.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "recordings")
SR = 12000
WIN = 10  # seconds

# A 10 s window is "signal" when enough of its 85 ms slices have a frequency standing
# well clear of the background, or when such slices come in a steady rhythm
# (buzz, pips). Tune with --dump on takes you have listened to.
PEAK_DB = 15.0     # a slice is "active" when its strongest frequency is this far up
ACTIVE_MIN = 0.20  # share of active slices that makes a window signal
PULSE_MIN = 0.40   # or: active slices recur this regularly (autocorrelation)
GOOD_FRACTION = 0.5   # take is "good" when this share of windows is signal
MAYBE_FRACTION = 0.15

# What each source is for in the mix.
BEDS = ("uvb76", "pip", "squeaky", "xpa", "xpb")
GROUPS = [
    ("beds: drones and markers", lambda n: n.startswith(BEDS)),
    ("voice numbers", lambda n: n[:1] in "esvg" and not n.startswith(BEDS)),
    ("morse", lambda n: n.startswith("m")),
    ("time signals", lambda n: n in ("wwv", "chu")),
]
TARGETS = {  # usable minutes wanted, from the 60-minute plan
    "beds: drones and markers": 120,
    "voice numbers": 60,
    "morse": 20,
    "time signals": 20,
}


def decode(path):
    """Any audio file -> mono float array at SR, via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
                          "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64)


def passband(a, nfft=2048):
    """Bins the receiver actually let through (CW filters are a few hundred Hz wide,
    so a fixed 300-3000 Hz band would mistake the filter edge for a tone)."""
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    step = max(nfft, len(a) // 200)
    segs = [a[j:j + nfft] * np.hanning(nfft) for j in range(0, len(a) - nfft, step)]
    spec = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], axis=0) + 1e-9
    db = 10 * np.log10(np.convolve(spec, np.ones(9) / 9, mode="same"))
    inside = (db >= np.percentile(db[(fr > 100) & (fr < 5000)], 90) - 15) & (fr > 100) & (fr < 5000)
    return inside


def periodicity(env, rate):
    """Best autocorrelation of the loudness curve at a 0.4-5 s lag: steady pulses score high."""
    e = env - env.mean()
    if not e.any():
        return 0.0
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac /= ac[0]
    lo, hi = int(0.4 * rate), min(len(ac) - 1, int(5 * rate))
    return float(ac[lo:hi].max()) if hi > lo else 0.0


FRAME = 1024          # 85 ms slices
FRAME_RATE = SR / (FRAME // 2)


def peak_series(a):
    """For every 85 ms slice: how far the strongest frequency stands above the slice's
    median, after dividing out the take's own background spectrum. Band noise and
    fading stay around 8-11 dB; tones, buzz harmonics, morse and voice jump above it.
    Constant interference lines are part of the background, so they don't count."""
    band = passband(a, FRAME)
    n = (len(a) - FRAME) // (FRAME // 2)
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(FRAME)[None, :] + (FRAME // 2) * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(a[idx] * np.hanning(FRAME), axis=1)) ** 2
    spec = spec[:, band] + 1e-9
    bg = np.median(spec, axis=0)  # the take's background, per frequency
    rel = spec / bg
    peak_db = 10 * np.log10(rel.max(axis=1) / np.median(rel, axis=1))
    loud = np.sqrt((a[idx] ** 2).mean(axis=1))
    return peak_db, loud


def window_features(a):
    peak_db, loud = peak_series(a)
    per = int(WIN * FRAME_RATE)
    out = []
    for k in range(0, len(peak_db) - per + 1, per):
        p, l = peak_db[k:k + per], loud[k:k + per]
        if l.mean() < 3:  # digital silence: receiver dropped out
            out.append({"t": int(k / FRAME_RATE), "active": 0.0, "peak": 0.0, "pulse": 0.0})
            continue
        out.append({"t": int(k / FRAME_RATE),
                    "active": float((p >= PEAK_DB).mean()),
                    "peak": float(np.percentile(p, 90)),
                    "pulse": periodicity(p, FRAME_RATE)})
    return out


def is_signal(f):
    return f["active"] >= ACTIVE_MIN or (f["pulse"] >= PULSE_MIN and f["active"] >= ACTIVE_MIN / 4)


def score_file(path):
    feats = window_features(decode(path))
    sig = [is_signal(f) for f in feats]
    n = len(feats)
    frac = sum(sig) / n if n else 0.0
    # strongest minute: the 6-window run with the most signal windows, then the highest tone
    best_t, best = 0, (-1, -1)
    for i in range(max(1, n - 5)):
        run = feats[i:i + 6]
        key = (sum(is_signal(f) for f in run), np.mean([f["peak"] for f in run]) if run else 0)
        if key > best:
            best, best_t = key, feats[i]["t"] if run else 0
    verdict = "good" if frac >= GOOD_FRACTION else "maybe" if frac >= MAYBE_FRACTION else "static"
    return {
        "minutes": round(n * WIN / 60, 1),
        "signal_minutes": round(sum(sig) * WIN / 60, 1),
        "signal_share": round(frac, 2),
        "verdict": verdict,
        "best_minute_at": f"{best_t // 60}:{best_t % 60:02d}",
        "median_peak_db": round(float(np.median([f["peak"] for f in feats])), 1) if feats else 0,
        "median_pulse": round(float(np.median([f["pulse"] for f in feats])), 2) if feats else 0,
    }


def take_info(path):
    """20261116T0159Z_v13_8169kHz_host.flac -> station, kHz, receiver, start."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    if len(parts) < 2 or not parts[0].endswith("Z"):
        return None
    return {"file": os.path.relpath(path, HERE), "start": parts[0], "station": parts[1],
            "khz": parts[2].replace("kHz", "") if len(parts) > 2 else "",
            "receiver": "_".join(parts[3:])}


FIELDS = ["file", "start", "station", "khz", "receiver", "minutes", "signal_minutes",
          "signal_share", "verdict", "best_minute_at", "median_peak_db", "median_pulse"]


def load_rows(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def report(rows):
    rows = sorted(rows.values(), key=lambda r: r["start"])
    by_group = {g: {"takes": 0, "good": 0, "minutes": 0.0, "stations": set()} for g, _ in GROUPS}
    for r in rows:
        for g, test in GROUPS:
            if test(r["station"]):
                s = by_group[g]
                s["takes"] += 1
                if r["verdict"] == "good":
                    s["good"] += 1
                if r["verdict"] in ("good", "maybe"):
                    s["minutes"] += float(r["signal_minutes"])
                    s["stations"].add(r["station"])
                break
    print(f"{len(rows)} takes scored, "
          f"{sum(r['verdict'] == 'good' for r in rows)} good, "
          f"{sum(r['verdict'] == 'maybe' for r in rows)} maybe, "
          f"{sum(r['verdict'] == 'static' for r in rows)} static\n")
    print(f"{'group':<26} {'takes':>5} {'good':>5} {'usable min':>11} {'target':>7}  stations heard")
    done = True
    for g, _ in GROUPS:
        s = by_group[g]
        target = TARGETS[g]
        mark = "ok" if s["minutes"] >= target else ""
        done &= s["minutes"] >= target
        print(f"{g:<26} {s['takes']:>5} {s['good']:>5} {s['minutes']:>11.0f} {target:>7}  {mark:<3}"
              f"{', '.join(sorted(s['stations']))}")
    print("\nEnough to stop recording." if done else "\nKeep recording: some groups are short of their target.")


def write_listen(rows, path, per_station=3):
    best = {}
    for r in rows.values():
        if r["verdict"] == "static":
            continue
        best.setdefault(r["station"], []).append(r)
    with open(path, "w") as f:
        f.write("Best takes per station (signal share, then strongest minute to jump to)\n\n")
        for st in sorted(best):
            f.write(f"{st}\n")
            for r in sorted(best[st], key=lambda r: -float(r["signal_share"]))[:per_station]:
                f.write(f"  {float(r['signal_share']):.0%}  at {r['best_minute_at']}  {r['file']}\n")
            f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--dump", nargs="+", metavar="FILE")
    args = p.parse_args()

    if args.dump:
        for path in args.dump:
            print(path)
            for f in window_features(decode(path)):
                print(f"  {f['t'] // 60:3d}:{f['t'] % 60:02d}  active {f['active']:.2f}  peak {f['peak']:5.1f} dB"
                      f"  pulse {f['pulse']:.2f}  {'SIGNAL' if is_signal(f) else ''}")
        return

    csv_path = os.path.join(REC, "triage.csv")
    rows = {} if args.rescore else load_rows(csv_path)
    # only finished takes in dated folders (the collector writes into kiwi-* temp folders first)
    files = sorted(glob.glob(os.path.join(REC, "????-??-??", "*.flac")) +
                   glob.glob(os.path.join(REC, "????-??-??", "*.wav")))
    todo = [f for f in files if take_info(f) and take_info(f)["file"] not in rows]
    for i, path in enumerate(todo, 1):
        info = take_info(path)
        print(f"[{i}/{len(todo)}] {info['file']}", file=sys.stderr, flush=True)
        try:
            info.update(score_file(path))
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"  skipped ({e})", file=sys.stderr)
            continue
        rows[info["file"]] = {k: info[k] for k in FIELDS}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: r["start"]):
            w.writerow(r)
    write_listen(rows, os.path.join(REC, "listen.txt"))
    report(rows)
    print(f"\nDetails: {os.path.relpath(csv_path, HERE)}   Listening list: recordings/listen.txt")


if __name__ == "__main__":
    main()
