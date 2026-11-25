# MesoXV

**A 15-minute mesoanalysis for storm chasing.** SPC's mesoanalysis refreshes every
hour from the RAP model and hourly surface observations. MesoXV refreshes the
surface every 15 minutes: NOAA's RTMA Rapid Update surface analysis (2.5 km)
underneath the latest HRRR profile, recomputed SPC-style on a 6 km grid.

Live page (when running): https://daandm87-cmyk.github.io/MesoXV/

## Maps

| Group | Maps |
|---|---|
| Surface | dewpoint, isobars, wind · moisture flux convergence · 1-hour pressure change |
| Instability | SB / ML (100 mb) / MU CAPE with CIN · 1-hour MLCAPE change · 1-hour MLCIN change (cap weakening) · ML LCL height · 700-500 mb lapse rate |
| Wind shear | 0-1 and 0-3 km SRH with Bunkers right-mover motion · 0-6 km bulk shear |
| Composites | STP (fixed layer) · SCP (fixed layer) |

US units (F, kt, J/kg, m2/s2, mb); times in US Central and UTC. Palettes avoid
red/green contrasts.

## How it works

1. **Surface, every 15 min:** RTMA Rapid Update analysis (2 m temperature and
   dewpoint, surface pressure, 10 m wind). It posts about 15-18 minutes after its
   valid time, so each frame is that far behind real time.
2. **Aloft, hourly:** HRRR pressure levels (25 mb spacing up to 850 mb, 50 mb above),
   interpolated in time between two forecast hours of the newest run that would
   actually be available at that moment.
3. **Column:** the RTMA surface replaces HRRR's, and the surface difference in
   potential temperature and moisture is mixed up through HRRR's boundary layer.
4. **Parcels and winds:** surface-based, 100 mb mixed-layer and most-unstable
   parcels (pseudo-adiabatic, virtual temperature); Bunkers right mover with SPC's
   convention (0-6 km non-pressure-weighted mean wind); SRH from the RTMA 10 m wind up.

Checked against MetPy on identical soundings (CAPE within 2-5 %, CIN and LCL height
essentially identical) and against HRRR's own CAPE fields (correlation 0.995-0.998).

**Differences from SPC's pages:** STP and SCP use the fixed-layer formulas (SPC's
headline maps use effective-layer versions), and HRRR rather than RAP supplies the
profile above the surface.

## Running it

GitHub Actions, on demand (the `MesoXV` workflow):

- **Live:** set `hours` (up to 12). A new frame every 15 minutes, the last 3 hours
  kept on the page. Jobs longer than ~5 hours hand over to a fresh job by themselves.
- **Replay:** set `replay` to a UTC window, e.g. `2026-04-17T19:00/2026-04-18T01:00`.

Locally:

```
pip install -r requirements.txt
python run.py --start 2026-04-17T19:00 --end 2026-04-18T01:00 --regions iowa,midwest
python run.py --live --hours 6
```

The site is written to `docs/` (open it through a local web server, e.g.
`python -m http.server` in that folder).

## Data

NOAA RTMA Rapid Update and HRRR from the NOAA Open Data Dissemination program on AWS
(`noaa-rtma-pds`, `noaa-hrrr-bdp-pds`). Map outlines: Natural Earth.
