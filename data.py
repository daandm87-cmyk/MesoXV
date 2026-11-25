"""Data access for MesoXV: partial GRIB2 downloads from the NOAA open-data
buckets on AWS (byte ranges picked from the .idx files), a disk cache, and
decoding with eccodes.

Sources (all verified 2026-11-24):
- RTMA Rapid Update, every 15 min, 2.5 km:  noaa-rtma-pds/rtma2p5_ru.YYYYMMDD/
  rtma2p5_ru.tHHMMz.2dvaranl_ndfd.grb2  (lands ~15-18 min after valid time)
- HRRR hourly runs: noaa-hrrr-bdp-pds/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrf{prs,sfc}fFF.grib2
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import os
import time
import urllib.error
import urllib.request

import numpy as np
import eccodes

RTMA_URL = "https://noaa-rtma-pds.s3.amazonaws.com/rtma2p5_ru.{d:%Y%m%d}/rtma2p5_ru.t{d:%H%M}z.2dvaranl_ndfd.grb2"
HRRR_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{r:%Y%m%d}/conus/hrrr.t{r:%H}z.wrf{kind}f{f:02d}.grib2"

CACHE = os.environ.get("MESOXV_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
UA = {"User-Agent": "MesoXV (storm chasing mesoanalysis)"}


class Missing(Exception):
    """The file (or its index) isn't on the server (yet)."""


def _get(url, headers=None, tries=4):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise Missing(url) from None
            if i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == tries - 1:
                raise
        time.sleep(2 * (i + 1))


def read_idx(url):
    """[(var, level, step, start, end_or_None)] from the .idx next to a GRIB2 file."""
    text = _get(url + ".idx").decode("ascii", "replace")
    rows = [l.split(":") for l in text.splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else None
        out.append((r[3], r[4], r[5], int(r[1]), end))
    return out


def _cache_path(url, var, level):
    rel = url.split(".amazonaws.com/")[1].replace("/", "_")
    safe = ("%s_%s" % (var, level)).replace(" ", "").replace("/", "-").replace(":", "")
    return os.path.join(CACHE, rel, safe + ".grib2")


def fetch(url, wanted, step=None):
    """Download the messages `wanted` [(var, level), ...] from one GRIB2 file.
    Returns {(var, level): path_to_single_message_file}. Cached on disk."""
    paths = {w: _cache_path(url, *w) for w in wanted}
    todo = [w for w in wanted if not os.path.exists(paths[w])]
    if todo:
        idx = read_idx(url)
        where = {}
        for var, level, stp, a, b in idx:
            if (var, level) in todo and (step is None or stp == step) and (var, level) not in where:
                where[(var, level)] = (a, b)
        lost = [w for w in todo if w not in where]
        if lost:
            raise Missing("%s: no %s" % (url, lost))

        def one(w):
            a, b = where[w]
            rng = "bytes=%d-%s" % (a, "" if b is None else b)
            data = _get(url, {"Range": rng})
            os.makedirs(os.path.dirname(paths[w]), exist_ok=True)
            tmp = paths[w] + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, paths[w])

        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(one, todo))
    return paths


def decode(path):
    """(values 2-D float32 with rows south->north, grid dict)"""
    with open(path, "rb") as fh:
        gid = eccodes.codes_grib_new_from_file(fh)
        try:
            nx = eccodes.codes_get(gid, "Nx")
            ny = eccodes.codes_get(gid, "Ny")
            vals = eccodes.codes_get_values(gid).astype(np.float32)
            miss = eccodes.codes_get(gid, "missingValue")
            grid = {k: eccodes.codes_get(gid, k) for k in (
                "gridType", "LoVInDegrees", "Latin1InDegrees", "Latin2InDegrees",
                "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
                "DxInMetres", "DyInMetres", "jScansPositively", "iScansNegatively",
                "uvRelativeToGrid")}
            grid.update(nx=nx, ny=ny)
            try:
                grid["radius"] = eccodes.codes_get(gid, "radius")
            except eccodes.KeyValueNotFoundError:
                grid["radius"] = 6371229.0
        finally:
            eccodes.codes_release(gid)
    vals[vals == miss] = np.nan
    a = vals.reshape(ny, nx)
    if not grid["jScansPositively"]:
        a = a[::-1]
    return a, grid


def load(url, wanted, step=None):
    """{(var, level): 2-D array}, plus the grid of the first message."""
    paths = fetch(url, wanted, step)
    out, grid = {}, None
    for w, p in paths.items():
        out[w], g = decode(p)
        grid = grid or g
    return out, grid


def rtma_url(valid: dt.datetime):
    return RTMA_URL.format(d=valid)


def hrrr_url(run: dt.datetime, fxx: int, kind="prs"):
    return HRRR_URL.format(r=run, f=fxx, kind=kind)
