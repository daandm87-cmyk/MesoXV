"""Compare MesoXV's vectorized parcels with MetPy on the same columns."""
import datetime as dt, warnings
import numpy as np
import frame, analysis as an
import metpy.calc as mpc
from metpy.units import units
warnings.filterwarnings("ignore")

valid = dt.datetime(2026, 4, 17, 21, 0)
inp = frame.Inputs((-104, -84, 36, 48), step=2, check=True)
run, hs, prof, check = inp.hrrr_at(valid)
s = dict(hs); s["t_model"], s["q_model"], s["p_model"] = hs["t"], hs["q"], hs["p"]
col = an.build_column(s, prof)
sb = an.parcel(col, col["p"][0], col["t"][0], col["q"][0])
ml = an.parcel(col, *an.ml_parcel_start(col))
rng = np.random.default_rng(1)
cand = np.argwhere(check["CAPE surface"] > 800)
pick = cand[rng.choice(len(cand), 12, replace=False)]
print("   j   i | SB mine  metpy  hrrr | SBCIN mine metpy | ML mine  metpy  hrrr90 | LCL mine metpy")
for j, i in pick:
    p = col["p"][:, j, i]; t = col["t"][:, j, i]; q = col["q"][:, j, i]; z = col["zagl"][:, j, i]
    keep = np.r_[True, p[1:] < p[0] - 0.5]          # drop the zero-thickness duplicates
    P = p[keep] * units.hPa; T = (t[keep] - 273.15) * units.degC
    Td = (an.td_from_q(q[keep], p[keep]) - 273.15) * units.degC
    c1, n1 = mpc.surface_based_cape_cin(P, T, Td)
    c2, n2 = mpc.mixed_layer_cape_cin(P, T, Td, depth=100 * units.hPa)
    lp, lt = mpc.lcl(P[0], T[0], Td[0])
    zl = np.interp(np.log(lp.m), np.log(p[keep][::-1]), z[keep][::-1])
    print("%4d %3d | %6.0f %6.0f %6.0f | %6.0f %6.0f | %6.0f %6.0f %6.0f | %5.0f %5.0f" % (
        j, i, sb[0][j, i], c1.m, check["CAPE surface"][j, i], sb[1][j, i], n1.m,
        ml[0][j, i], c2.m, check["CAPE 90-0 mb above ground"][j, i], sb[3][j, i], zl))
