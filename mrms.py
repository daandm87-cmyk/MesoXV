"""Radar context for the maps: MRMS composite reflectivity (every 2 min, 1 km),
max-pooled onto the analysis grid so small cores survive the 6 km thinning.
Source: noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/YYYYMMDD/ (gzipped GRIB2).
"""
from __future__ import annotations

import datetime as dt
import gzip
import os
import re
import urllib.parse

import numpy as np
import eccodes
from scipy.ndimage import maximum_filter, map_coordinates

import data

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "MergedReflectivityQCComposite_00.50"
MAX_AGE = dt.timedelta(minutes=10)


def _key_for(valid):
    """Newest file at or before `valid` (within MAX_AGE)."""
    best = None
    for day, h in sorted({(t.strftime("%Y%m%d"), t.strftime("%H")) for t in (valid - MAX_AGE, valid)}):
        stem = "CONUS/%s/%s/MRMS_%s_%s-" % (PRODUCT, day, PRODUCT, day)
        xml = data._get("%s/?list-type=2&prefix=%s" % (BUCKET, urllib.parse.quote(stem + h))).decode("utf-8", "replace")
        for key in re.findall(r"<Key>([^<]+)</Key>", xml):
            t = dt.datetime.strptime(key[-24:-9], "%Y%m%d-%H%M%S")
            if valid - MAX_AGE <= t <= valid + dt.timedelta(seconds=59) and (best is None or t > best[0]):
                best = (t, key)
    return best


def reflectivity(valid, grid):
    """Composite reflectivity (dBZ) on the analysis grid, or None if there's no file."""
    got = _key_for(valid)
    if got is None:
        return None
    t, key = got
    path = os.path.join(data.CACHE, "mrms", os.path.basename(key))
    if not os.path.exists(path):
        raw = data._get("%s/%s" % (BUCKET, key))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
    msg = gzip.open(path).read()
    gid = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lo1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        di = eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
        dj = eccodes.codes_get(gid, "jDirectionIncrementInDegrees")
        north_first = not eccodes.codes_get(gid, "jScansPositively")
        z = eccodes.codes_get_values(gid).astype(np.float32).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    z[z < 0] = 0.0                                   # -999 no coverage, -99 no echo
    lo1 = lo1 - 360.0 if lo1 > 180 else lo1
    # only the part of the 7000 x 3500 grid the analysis covers, then max-pool ~6 km
    lat, lon = grid.lat, grid.lon
    fi = (lon - lo1) / di
    fj = (la1 - lat) / dj if north_first else (lat - la1) / dj
    i0, i1 = max(int(fi.min()) - 8, 0), min(int(fi.max()) + 9, ni)
    j0, j1 = max(int(fj.min()) - 8, 0), min(int(fj.max()) + 9, nj)
    sub = maximum_filter(z[j0:j1, i0:i1], size=5)
    out = map_coordinates(sub, [(fj - j0).ravel(), (fi - i0).ravel()], order=0, mode="nearest")
    return out.reshape(grid.shape).astype(np.float32), t
