#!/usr/bin/env python3
"""Collect source recordings for the OO Radio mix.

Records shortwave stations (numbers stations, time signals, marker buzzers)
through public KiwiSDR receivers, and ordinary internet audio streams through
ffmpeg. Timed stations come from schedule.txt; always-on stations and streams
come from sources.json and are sampled every few hours.

Commands:
  python3 collect.py setup                 fetch kiwiclient (needs git; numpy for running it)
  python3 collect.py receivers [--khz F] [--near LAT,LON]
                                           list free public KiwiSDRs, nearest first
  python3 collect.py plan [--hours 24]     show what will be recorded and when (UTC)
  python3 collect.py record NAME [--minutes M]
                                           record one source right now
  python3 collect.py run                   keep running and record everything on schedule

Only the Python standard library is used here. Recording shortwave needs
kiwiclient (see `setup`), which needs numpy. Recording streams needs ffmpeg.
"""

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = dt.timezone.utc
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
KIWI_LIST_URL = "http://rx.linkfanel.net/kiwisdr_com.js"
KIWI_CACHE = os.path.join(HERE, ".kiwi-list-cache.json")
KIWICLIENT_DIR = os.path.join(HERE, "kiwiclient")
USER_TAG = "oo-radio-collect"

log_lock = threading.Lock()


def say(msg):
    stamp = dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    with log_lock:
        print(f"[{stamp}] {msg}", flush=True)


# ---------------------------------------------------------------- config

def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("out_dir", "recordings")
    cfg.setdefault("lead_seconds", 60)
    cfg.setdefault("tail_seconds", 60)
    cfg.setdefault("receivers_to_try", 4)
    cfg.setdefault("near", [52.0, 10.0])
    cfg.setdefault("receivers", [])
    cfg.setdefault("sources", [])
    if not os.path.isabs(cfg["out_dir"]):
        cfg["out_dir"] = os.path.join(HERE, cfg["out_dir"])
    return cfg


def parse_hhmm(s):
    h, m = s.split(":")
    return int(h), int(m)


def parse_days(s):
    if not s or s in ("daily", "*"):
        return set(range(7))
    out = set()
    for part in s.lower().split(","):
        if "-" in part:
            a, b = (DAYS.index(x[:3]) for x in part.split("-"))
            i = a
            while True:
                out.add(i)
                if i == b:
                    break
                i = (i + 1) % 7
        else:
            out.add(DAYS.index(part[:3]))
    return out


def load_schedule(path):
    """Lines: HH:MM  minutes  freq_khz  mode  name  [days]  [# note]"""
    items = []
    if not os.path.exists(path):
        return items
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                say(f"schedule.txt line {n} skipped (need: HH:MM minutes kHz mode name [days])")
                continue
            try:
                h, m = parse_hhmm(parts[0])
                items.append({
                    "name": parts[4],
                    "type": "kiwi",
                    "utc": (h, m),
                    "minutes": float(parts[1]),
                    "freq_khz": float(parts[2]),
                    "mode": parts[3].lower(),
                    "days": parse_days(parts[5] if len(parts) > 5 else "daily"),
                })
            except (ValueError, IndexError) as e:
                say(f"schedule.txt line {n} skipped ({e})")
    return items


def freq_for(source, when):
    """Pick the frequency for a source at a UTC time (supports day/night pairs)."""
    if "freq_khz" in source:
        return source["freq_khz"]
    hour = when.hour + when.minute / 60
    for f in source["freqs"]:
        a, b = (int(x) for x in f.get("utc", "0-24").split("-"))
        inside = a <= hour < b if a <= b else (hour >= a or hour < b)
        if inside:
            return f["khz"]
    return source["freqs"][0]["khz"]


# ---------------------------------------------------------------- planning

def upcoming(cfg, schedule, start, hours):
    """Every recording that should start between start and start+hours."""
    end = start + dt.timedelta(hours=hours)
    jobs = []
    lead = dt.timedelta(seconds=cfg["lead_seconds"])
    tail = cfg["tail_seconds"]

    for item in schedule:
        day = (start - dt.timedelta(days=1)).date()
        while day <= end.date():
            t = dt.datetime(day.year, day.month, day.day, *item["utc"], tzinfo=UTC)
            rec_start = t - lead
            if t.weekday() in item["days"] and start <= rec_start < end:
                secs = item["minutes"] * 60 + cfg["lead_seconds"] + tail
                jobs.append((rec_start, item, item["freq_khz"], secs))
            day += dt.timedelta(days=1)

    for src in cfg["sources"]:
        if not src.get("enabled", True):
            continue
        every = dt.timedelta(hours=src.get("every_hours", 6))
        offset = dt.timedelta(minutes=src.get("offset_minutes", 0))
        anchor = dt.datetime(start.year, start.month, start.day, tzinfo=UTC) + offset
        t = anchor
        while t < end:
            if t >= start:
                freq = freq_for(src, t) if src["type"] == "kiwi" else None
                jobs.append((t, src, freq, src.get("minutes", 10) * 60))
            t += every

    jobs.sort(key=lambda j: j[0])
    return jobs


# ---------------------------------------------------------------- receivers

def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def fetch_kiwi_list(max_age_hours=6):
    if os.path.exists(KIWI_CACHE) and time.time() - os.path.getmtime(KIWI_CACHE) < max_age_hours * 3600:
        with open(KIWI_CACHE) as f:
            return json.load(f)
    req = urllib.request.Request(KIWI_LIST_URL, headers={"User-Agent": USER_TAG})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", "replace")
    rows = parse_kiwi_list(text)
    with open(KIWI_CACHE, "w") as f:
        json.dump(rows, f)
    return rows


def parse_kiwi_list(text):
    """Parse the public KiwiSDR directory (a JS array of objects)."""
    body = text[text.find("["): text.rfind("]") + 1]
    try:
        raw = json.loads(body)
    except ValueError:
        # tolerate trailing commas and other JS looseness
        raw = []
        for obj in re.findall(r"\{[^{}]*\}", body, re.S):
            try:
                raw.append(json.loads(re.sub(r",\s*}", "}", obj)))
            except ValueError:
                pass
    rows = []
    for r in raw:
        url = r.get("url", "")
        m = re.match(r"https?://([^/:]+)(?::(\d+))?", url)
        gps = re.findall(r"-?\d+(?:\.\d+)?", str(r.get("gps", "")))
        if not m or len(gps) < 2:
            continue
        if str(r.get("offline", "no")).lower() == "yes" or str(r.get("status", "active")).lower() not in ("active", ""):
            continue
        lo_hi = re.findall(r"\d+", str(r.get("bands", "0-30000000")))
        rows.append({
            "host": m.group(1),
            "port": int(m.group(2) or 8073),
            "name": r.get("name", ""),
            "lat": float(gps[0]),
            "lon": float(gps[1]),
            "users": int(r.get("users", 0) or 0),
            "users_max": int(r.get("users_max", 4) or 4),
            "band_lo_khz": int(lo_hi[0]) / 1000 if lo_hi else 0,
            "band_hi_khz": int(lo_hi[1]) / 1000 if len(lo_hi) > 1 else 30000,
        })
    return rows


def pick_receivers(cfg, source, freq_khz):
    """Fixed receivers first (from config), then nearest free public ones."""
    fixed = source.get("receivers") or cfg["receivers"]
    out = []
    for r in fixed:
        host, _, port = r.partition(":")
        out.append((host, int(port or 8073)))
    if source.get("receivers_only"):
        return out
    near = source.get("near", cfg["near"])
    try:
        rows = fetch_kiwi_list()
    except Exception as e:
        say(f"could not read public KiwiSDR list ({e}); using fixed receivers only")
        return out
    rows = [r for r in rows
            if r["users"] < r["users_max"] and r["band_lo_khz"] <= freq_khz <= r["band_hi_khz"]]
    rows.sort(key=lambda r: haversine_km(near, (r["lat"], r["lon"])))
    for r in rows:
        if (r["host"], r["port"]) not in out:
            out.append((r["host"], r["port"]))
        if len(out) >= cfg["receivers_to_try"] + len(fixed):
            break
    return out


# ---------------------------------------------------------------- recording

def out_path(cfg, name, when, extra):
    day_dir = os.path.join(cfg["out_dir"], when.strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)
    base = f"{when.strftime('%Y%m%dT%H%MZ')}_{name}"
    if extra:
        base += f"_{extra}"
    return os.path.join(day_dir, base + ".wav")


def write_log(cfg, entry):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with log_lock, open(os.path.join(cfg["out_dir"], "log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def record_kiwi(cfg, source, freq_khz, seconds, when):
    script = os.path.join(KIWICLIENT_DIR, "kiwirecorder.py")
    if not os.path.exists(script):
        raise SystemExit("kiwiclient is missing: run `python3 collect.py setup` first")
    mode = source.get("mode", "usb")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    for host, port in pick_receivers(cfg, source, freq_khz):
        tmp = tempfile.mkdtemp(prefix="kiwi-", dir=cfg["out_dir"])
        cmd = [sys.executable, script, "-s", host, "-p", str(port),
               "-f", f"{freq_khz:g}", "-m", mode, "--tlimit", str(int(seconds)),
               "-d", tmp, "--fn", "take", "-u", USER_TAG, "-q",
               "--connect-retries", "1", "--busy-retries", "1",
               "--connect-timeout", "10", "--busy-timeout", "5"]
        if "low_hz" in source:
            cmd += ["-L", str(source["low_hz"]), "-H", str(source["high_hz"])]
        say(f"{source['name']}: {freq_khz:g} kHz {mode} on {host}:{port} for {seconds / 60:.1f} min")
        try:
            subprocess.run(cmd, timeout=seconds + 120, cwd=KIWICLIENT_DIR,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            pass
        wavs = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".wav")]
        got = max(wavs, key=os.path.getsize) if wavs else None
        size = os.path.getsize(got) if got else 0
        # 12 kHz, 16-bit mono = 24 kB/s; demand at least a quarter of the asked length
        if got and size > 24000 * seconds * 0.25:
            dest = out_path(cfg, source["name"], when, f"{freq_khz:g}kHz_{host}")
            shutil.move(got, dest)
            shutil.rmtree(tmp, ignore_errors=True)
            say(f"{source['name']}: saved {os.path.relpath(dest, HERE)} ({size / 1e6:.1f} MB)")
            return {"file": dest, "receiver": f"{host}:{port}", "bytes": size}
        shutil.rmtree(tmp, ignore_errors=True)
        say(f"{source['name']}: {host}:{port} gave nothing usable, trying next receiver")
    return None


def record_stream(cfg, source, seconds, when):
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not installed")
    dest = out_path(cfg, source["name"], when, "")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if source["url"].startswith("http"):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
    cmd += ["-i", source["url"], "-t", str(int(seconds)), "-vn", "-ac", "2", "-ar", "44100", dest]
    say(f"{source['name']}: stream for {seconds / 60:.1f} min")
    try:
        subprocess.run(cmd, timeout=seconds + 120, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        pass
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        say(f"{source['name']}: saved {os.path.relpath(dest, HERE)}")
        return {"file": dest, "receiver": source["url"], "bytes": os.path.getsize(dest)}
    say(f"{source['name']}: stream gave nothing")
    return None


def run_job(cfg, source, freq, seconds):
    when = dt.datetime.now(UTC)
    entry = {"name": source["name"], "label": source.get("label", ""),
             "start_utc": when.isoformat(timespec="seconds"), "seconds": seconds,
             "type": source["type"], "freq_khz": freq, "mode": source.get("mode"),
             "credit": source.get("credit", "")}
    try:
        if source["type"] == "kiwi":
            result = record_kiwi(cfg, source, freq, seconds, when)
        else:
            result = record_stream(cfg, source, seconds, when)
    except Exception as e:  # keep the scheduler alive
        say(f"{source['name']}: failed ({e})")
        result = None
    entry.update(result or {"file": None})
    entry["ok"] = bool(result)
    write_log(cfg, entry)
    return result


# ---------------------------------------------------------------- commands

def cmd_setup(cfg, args):
    if os.path.exists(KIWICLIENT_DIR):
        print("kiwiclient already present:", KIWICLIENT_DIR)
    else:
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/jks-prv/kiwiclient", KIWICLIENT_DIR], check=True)
    try:
        import numpy  # noqa: F401
        print("numpy: ok")
    except ImportError:
        print("numpy is missing: run `python3 -m pip install numpy`")
    print("ffmpeg:", shutil.which("ffmpeg") or "missing (needed only for internet streams)")


def cmd_receivers(cfg, args):
    near = [float(x) for x in args.near.split(",")] if args.near else cfg["near"]
    rows = fetch_kiwi_list(max_age_hours=0)
    if args.khz:
        rows = [r for r in rows if r["band_lo_khz"] <= args.khz <= r["band_hi_khz"]]
    rows = [r for r in rows if r["users"] < r["users_max"]]
    rows.sort(key=lambda r: haversine_km(near, (r["lat"], r["lon"])))
    for r in rows[: args.limit]:
        km = haversine_km(near, (r["lat"], r["lon"]))
        print(f"{r['host']}:{r['port']:<6} {km:6.0f} km  {r['users']}/{r['users_max']} users  {r['name'][:60]}")


def cmd_plan(cfg, args):
    schedule = load_schedule(os.path.join(HERE, args.schedule))
    now = dt.datetime.now(UTC)
    for t, src, freq, secs in upcoming(cfg, schedule, now, args.hours):
        what = f"{freq:g} kHz {src.get('mode', '')}" if freq else src.get("url", "")
        print(f"{t.strftime('%a %m-%d %H:%M')}Z  {secs / 60:5.1f} min  {src['name']:<16} {what}")


def find_source(cfg, schedule, name):
    for s in cfg["sources"] + schedule:
        if s["name"] == name:
            return s
    raise SystemExit(f"no source called {name!r} in sources.json or schedule.txt")


def cmd_record(cfg, args):
    schedule = load_schedule(os.path.join(HERE, args.schedule))
    src = find_source(cfg, schedule, args.name)
    minutes = args.minutes or src.get("minutes", 10)
    freq = freq_for(src, dt.datetime.now(UTC)) if src["type"] == "kiwi" else None
    ok = run_job(cfg, src, freq, minutes * 60)
    sys.exit(0 if ok else 1)


def cmd_run(cfg, args):
    say(f"collector running; recordings go to {cfg['out_dir']}")
    started = set()
    while True:
        schedule = load_schedule(os.path.join(HERE, args.schedule))  # re-read so edits apply
        now = dt.datetime.now(UTC)
        jobs = upcoming(cfg, schedule, now - dt.timedelta(seconds=30), 2)
        for t, src, freq, secs in jobs:
            key = (src["name"], t.isoformat())
            if key in started or t > now:
                continue
            started.add(key)
            threading.Thread(target=run_job, args=(cfg, src, freq, secs), daemon=True).start()
        future = [j[0] for j in jobs if j[0] > now]
        wait = (min(future) - now).total_seconds() if future else 600
        time.sleep(max(1, min(wait, 600)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="sources.json")
    p.add_argument("--schedule", default="schedule.txt")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("receivers")
    r.add_argument("--khz", type=float)
    r.add_argument("--near")
    r.add_argument("--limit", type=int, default=15)
    pl = sub.add_parser("plan")
    pl.add_argument("--hours", type=float, default=24)
    rec = sub.add_parser("record")
    rec.add_argument("name")
    rec.add_argument("--minutes", type=float)
    sub.add_parser("run")
    args = p.parse_args()
    cfg = load_config(os.path.join(HERE, args.config))
    {"setup": cmd_setup, "receivers": cmd_receivers, "plan": cmd_plan,
     "record": cmd_record, "run": cmd_run}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
