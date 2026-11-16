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
- **priyom.org schedule** (`"priyom"` in `sources.json`): while `run` is going, timed numbers-station transmissions are read from priyom.org's live calendar (refreshed every 6 hours) and each is recorded for `minutes`. `only` limits it to station families by prefix; the default keeps voice (E, S, V, G), morse (M) and polytone (XP) stations and skips the F and P data modems. Stations priyom marks as aimed at East Asia, North America, the Middle East or the Pacific are recorded from receivers in that region.
- **`schedule.txt`**: extra timed transmissions of your own, one per line: `HH:MM minutes kHz mode name [days]`. Recording starts a minute early and ends a minute late.

Finished takes are converted to FLAC (`"flac": false` keeps WAV). Shortwave noise doesn't compress well, so budget about 1.5 GB a day with the priyom schedule on.

## Sorting the takes

```sh
python3 triage.py            # score new takes, show progress against the mix targets
python3 triage.py --rescore  # score everything again after changing thresholds
python3 triage.py --dump recordings/2026-11-16/some_take.flac   # per-10 s numbers, for tuning
```

Each take is marked good, maybe or static by how much of it has a real signal standing out of the band noise. The report adds up usable minutes per group (beds: buzzers, pips and polytones; voice numbers; morse; time signals) against targets for a 60-minute mix and says when there's enough to stop recording. `recordings/listen.txt` lists the best takes per station with the minute to jump to; `recordings/triage.csv` has every take. The thresholds and targets sit at the top of `triage.py`.

## Receivers

For each recording the script picks free public KiwiSDRs nearest the source's `near` point and falls back to the next one if a receiver is busy or silent. To use receivers you've found sound good, list them as `"receivers": ["host:port"]` (globally or per source); they're tried first. Kiwis are volunteer-run with a few slots each, so keep takes short (10–15 minutes) and don't record from several at once.
