# OO Radio source collector

Gathers raw recordings for the Outside Observations Radio mix: numbers stations, marker buzzers and time signals through public KiwiSDR shortwave receivers, plus ordinary internet streams through ffmpeg.

## Setup (once)

```sh
git clone https://github.com/yoshisodeoka/oo-radio && cd oo-radio
python3 -m pip install numpy        # needed by kiwiclient
python3 collect.py setup            # downloads kiwiclient next to this script
```

You also need ffmpeg for internet streams (`brew install ffmpeg` on a Mac).

## Use

```sh
python3 collect.py plan             # what will be recorded in the next 24 h (UTC)
python3 collect.py record uvb76 --minutes 3   # quick test, records right now
python3 collect.py run              # leave running; records everything on schedule
python3 collect.py receivers --khz 4625 --near 55,30   # free receivers nearest a point
```

Recordings land in `recordings/YYYY-MM-DD/`, named by time, station, frequency and receiver. Every attempt is appended to `recordings/log.jsonl` for notes and credits.

## What gets recorded

- **`sources.json`**: stations that are always on (UVB-76, The Pip, The Squeaky Wheel, WWV, CHU), sampled every few hours. Each source has `every_hours`, `minutes` and `offset_minutes`. There are also two disabled stream slots: paste a NOAA Weather Radio URL (wxradio.org) or a Locus Sonus open-mic URL (locusonus.org/soundmap) and set `"enabled": true`.
- **`schedule.txt`**: timed voice and morse transmissions copied from the priyom.org schedule, one per line: `HH:MM minutes kHz mode name [days]`. Recording starts a minute early and ends a minute late.

## Receivers

For each recording the script picks free public KiwiSDRs nearest the source's `near` point and falls back to the next one if a receiver is busy or silent. To use receivers you've found sound good, list them as `"receivers": ["host:port"]` (globally or per source); they're tried first. Kiwis are volunteer-run with a few slots each, so keep takes short (10–15 minutes) and don't record from several at once.
