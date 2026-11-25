"""Observed low-level wind shear from NEXRAD VAD wind profiles (Level III NVW).

Each WSR-88D measures a wind profile every volume scan (~4-10 min). The NVW
product carries a plain-text "VAD Algorithm Output" table (altitude in 100 ft
MSL, u/v in m/s, earth-relative), read straight from the file.

Observed 0-1 km SRH = RTMA 10 m wind at the radar + the VAD levels up to 1 km,
with MesoXV's Bunkers storm motion at that point. Comparing it with MesoXV's own
0-1 km SRH there shows how far the model's low-level winds are off.
Source: unidata-nexrad-level3 on AWS (keys SITE_NVW_YYYY_MM_DD_HH_MM_SS).
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import time
import urllib.parse

import numpy as np

import data
import grids

BUCKET = "https://unidata-nexrad-level3.s3.amazonaws.com"
HERE = os.path.dirname(os.path.abspath(__file__))
RADARS = json.load(open(os.path.join(HERE, "radars.json"), encoding="utf-8"))
MAX_AGE = dt.timedelta(minutes=12)      # profile at most this much older than the frame
MIN_TOP_M = 850.0                       # need VAD winds up to at least this height AGL
ROW = re.compile(r"(\d{3})\s+(-?\d+\.\d|NA)\s+(-?\d+\.\d|NA)\s+(-?\d+\.\d|NA)\s+(\d{3})\s+(\d{3})")

_lists = {}


def _keys(site, hour, valid):
    """Object keys of one radar-hour (re-listed while the hour is still filling)."""
    k = (site, hour)
    got = _lists.get(k)
    if got is not None:
        keys, at = got
        newest = max((_key_time(x) for x in keys), default=None)
        if (newest is not None and newest >= valid) or time.time() - at < 60 or \
                hour + dt.timedelta(hours=1, minutes=15) < dt.datetime.now(dt.timezone.utc).replace(tzinfo=None):
            return keys
    prefix = "%s_NVW_%s" % (site, hour.strftime("%Y_%m_%d_%H"))
    xml = data._get("%s/?list-type=2&prefix=%s" % (BUCKET, urllib.parse.quote(prefix))).decode("utf-8", "replace")
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    _lists[k] = (keys, time.time())
    return keys


def _key_time(key):
    return dt.datetime.strptime(key[-19:], "%Y_%m_%d_%H_%M_%S")


def latest(site, valid):
    """(time, [(alt_m_msl, u, v), ...]) of the newest profile in [valid - MAX_AGE, valid + 3 min]."""
    best = None
    for h in sorted({(valid - MAX_AGE).replace(minute=0, second=0), valid.replace(minute=0, second=0)}):
        for key in _keys(site, h, valid):
            t = _key_time(key)
            if valid - MAX_AGE <= t <= valid + dt.timedelta(minutes=3) and (best is None or t > best[0]):
                best = (t, key)
    if best is None:
        return None
    path = os.path.join(data.CACHE, "nvw", best[1])
    if not os.path.exists(path):
        raw = data._get("%s/%s" % (BUCKET, best[1]))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
    text = open(path, "rb").read().decode("latin-1")
    i = text.find("VAD Algorithm Output")
    if i < 0:
        return None
    rows = []
    for alt, u, v, _w, _d, _s in ROW.findall(text[i:i + 6000]):
        if u != "NA" and v != "NA":
            rows.append((int(alt) * 100 * 0.3048, float(u), float(v)))
    return best[0], rows


def _srh_1d(h, u, v, cu, cv, depth=1000.0):
    hs = np.arange(0.0, depth + 1, 100.0)
    ui, vi = np.interp(hs, h, u), np.interp(hs, h, v)
    su, sv = ui - cu, vi - cv
    return float((su[1:] * sv[:-1] - su[:-1] * sv[1:]).sum())


def observe(valid, grid, d):
    """Observed vs analysed 0-1 km SRH at every radar inside the analysis grid."""
    p = grids.proj_of(grid.hgrid)
    inside = []
    for r in RADARS:
        x, y = p(r["lon"], r["lat"])
        i = int(round((x - grid.x[0]) / grid.dx))
        j = int(round((y - grid.y[0]) / grid.dx))
        if 1 <= i < grid.shape[1] - 1 and 1 <= j < grid.shape[0] - 1:
            inside.append((r, x, y, j, i))

    def one(item):
        r, x, y, j, i = item
        try:
            got = latest(r["id"], valid)
        except Exception:
            return None
        if not got:
            return None
        t, rows = got
        zs = float(d["zs"][j, i])
        lev = sorted((alt - zs, u, v) for alt, u, v in rows if 0 < alt - zs <= 1600)
        if len([l for l in lev if l[0] <= 1000]) < 2 or max((l[0] for l in lev), default=0) < MIN_TOP_M:
            return None
        h = np.array([0.0] + [l[0] for l in lev])
        ue = np.array([l[1] for l in lev])
        ve = np.array([l[2] for l in lev])
        ug, vg = grids.earth_to_grid(ue, ve, grid.hgrid, np.full(ue.shape, r["lon"]))
        u = np.r_[d["u10"][j, i], ug]
        v = np.r_[d["v10"][j, i], vg]
        obs = _srh_1d(h, u, v, float(d["rm_u"][j, i]), float(d["rm_v"][j, i]))
        ana = float(d["srh1"][j, i])
        return {"site": r["id"], "lat": r["lat"], "lon": r["lon"], "x": float(x), "y": float(y), "obs": round(obs), "ana": round(ana),
                "diff": round(obs - ana), "age_min": max(0, int((valid - t).total_seconds() // 60))}

    with cf.ThreadPoolExecutor(16) as ex:
        return [o for o in ex.map(one, inside) if o]
