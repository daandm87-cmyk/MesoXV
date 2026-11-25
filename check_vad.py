"""Diagnose radar-vs-MesoXV 0-1 km SRH at chosen radars: per frame, the VAD levels
against MesoXV's own winds at the same heights.

python check_vad.py 2026-04-17T19:00 2026-04-18T01:00 UEX,LNX,OAX
"""
import datetime as dt
import sys
import warnings

import numpy as np

import analysis as an
import frame
import grids
import vad

warnings.filterwarnings("ignore")
start, end = (dt.datetime.fromisoformat(a) for a in sys.argv[1:3])
sites = sys.argv[3].split(",")
inp = frame.Inputs((-104, -84, 36, 48), step=2)


def wdir(u, v):
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360


t = start
while t <= end:
    d = frame.make_frame(inp, t)
    g = inp.grid
    _, hs, prof, _ = inp.hrrr_at(t)
    for o in d["_vad"]:
        if o["site"] not in sites:
            continue
        r = next(x for x in vad.RADARS if x["id"] == o["site"])
        i = int(round((o["x"] - g.x[0]) / g.dx))
        j = int(round((o["y"] - g.y[0]) / g.dx))
        print("\n%s %s  obs %d  MesoXV %d  diff %+d  (profile %d min old)  storm motion %.0f kt from %03.0f" % (
            t.strftime("%H:%MZ"), o["site"], o["obs"], o["ana"], o["diff"], o["age_min"],
            np.hypot(d["rm_u"][j, i], d["rm_v"][j, i]) * 1.944, wdir(d["rm_u"][j, i], d["rm_v"][j, i])))
        _, rows = vad.latest(o["site"], t)
        zs = float(d["zs"][j, i])
        # MesoXV (HRRR) wind at the same heights, grid-relative -> earth
        zl = prof["z"][:, j, i] - zs
        order = np.argsort(zl)
        ue_m, ve_m = grids.grid_to_earth(prof["u"][:, j, i], prof["v"][:, j, i], g.hgrid, np.full(zl.shape, r["lon"]))
        u10e, v10e = grids.grid_to_earth(d["u10"][j, i], d["v10"][j, i], g.hgrid, r["lon"])
        print("   10 m  RTMA %3.0f kt from %03.0f" % (np.hypot(u10e, v10e) * 1.944, wdir(u10e, v10e)))
        for alt, u, v in rows:
            h = alt - zs
            if 0 < h <= 1300:
                um = np.interp(h, zl[order], ue_m[order]); vm = np.interp(h, zl[order], ve_m[order])
                print("   %4.0f m  radar %3.0f kt from %03.0f   |  HRRR %3.0f kt from %03.0f" % (
                    h, np.hypot(u, v) * 1.944, wdir(u, v), np.hypot(um, vm) * 1.944, wdir(um, vm)))
    t += dt.timedelta(minutes=15)
