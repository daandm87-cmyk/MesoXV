"""Sanity check: MesoXV's parcel/wind code on HRRR's own surface vs HRRR's own fields."""
import datetime as dt, sys, time
import numpy as np
import frame, grids

valid = dt.datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-04-17T21:00")
inp = frame.Inputs((-104, -84, 36, 48), step=2, check=True)
t = time.time()
m = frame.make_frame(inp, valid, model_only=True)
print("frame in %.1f s, grid %s, run %s" % (time.time() - t, inp.grid.shape, m["_run"]))
c = m["_check"]

def cmp(name, mine, ref, mask=None):
    ok = np.isfinite(mine) & np.isfinite(ref)
    if mask is not None:
        ok &= mask
    a, b = mine[ok], ref[ok]
    r = np.corrcoef(a, b)[0, 1]
    print("%-22s r=%.3f  mean mine %8.1f  hrrr %8.1f  p90 mine %8.1f  hrrr %8.1f  n=%d"
          % (name, r, a.mean(), b.mean(), np.percentile(a, 90), np.percentile(b, 90), ok.sum()))

cmp("SBCAPE", m["sbcape"], c["CAPE surface"])
cmp("MLCAPE (100 vs 90mb)", m["mlcape"], c["CAPE 90-0 mb above ground"])
cmp("MUCAPE", m["mucape"], c["CAPE 255-0 mb above ground"])
cape_ok = (c["CAPE surface"] > 250)
cmp("SBCIN (CAPE>250)", np.nan_to_num(m["sbcin"]), c["CIN surface"], cape_ok)
cmp("MLCIN (CAPE>250)", np.nan_to_num(m["mlcin"]), c["CIN 90-0 mb above ground"], c["CAPE 90-0 mb above ground"] > 250)
cmp("SRH 0-1", m["srh1"], c["HLCY 1000-0 m above ground"])
cmp("SRH 0-3", m["srh3"], c["HLCY 3000-0 m above ground"])

# wind rotation: RTMA rotated into the HRRR grid vs HRRR's own 10 m wind
r = inp.rtma(valid)
_, hs, _, _ = inp.hrrr_at(valid)
def angdiff(u1, v1, u2, v2):
    d = np.degrees(np.arctan2(v1, u1) - np.arctan2(v2, u2))
    return (d + 180) % 360 - 180
strong = np.hypot(hs["u"], hs["v"]) > 5
d = angdiff(r["u"], r["v"], hs["u"], hs["v"])[strong]
print("RTMA vs HRRR 10 m wind direction: median diff %.1f deg, |diff| median %.1f" % (np.median(d), np.median(np.abs(d))))
