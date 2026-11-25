"""MesoXV analysis: one 15-minute frame.

Column = RTMA surface (2 m T/Td, surface pressure, 10 m wind) underneath the
HRRR pressure-level profile (time-interpolated between two forecast hours).
The surface difference RTMA - HRRR (potential temperature and moisture) is
mixed up through HRRR's boundary layer, the way an observed surface
modifies the model's mixed layer. Above that the profile is pure HRRR.

All arrays are (level, y, x) or (y, x) on the analysis grid. Level 0 is the
surface; pressure levels at or below the ground are replaced by the surface
values, so they form zero-thickness layers and drop out of every integral.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

RD, RV, CP, G = 287.04, 461.5, 1005.7, 9.80665
EPS = RD / RV
KAPPA = RD / CP

# ------------------------------------------------------------------ tunables
ML_DEPTH_HPA = 100.0         # mixed-layer parcel: lowest 100 hPa (SPC)
MU_DEPTH_HPA = 300.0         # most-unstable parcel searched in the lowest 300 hPa (SPC)
PBL_MIN_M = 150.0            # surface correction reaches at least this high
PBL_TAPER = 0.5              # ...and fades out over another half PBL depth above it

# Tornado visibility: Daan's LP / Classic / HP index from Tornado Visibility
# (C:\Tornado_Visibility\chase_beauty.py). Keep these in step with that file.
TV_SR_FLOOR_KT, TV_SR_CEIL_KT = 40.0, 60.0       # anvil (300 mb) storm-relative wind: HP -> LP
TV_LLSR_FLOOR_KT, TV_LLSR_CEIL_KT = 20.0, 50.0   # 10 m storm-relative inflow: second pathway
TV_PW_NEUTRAL_IN, TV_PW_DRY_IN, TV_PW_WET_IN, TV_PW_WEIGHT = 1.5, 1.0, 0.5, 0.12
TV_CA_OROG_MAX = 1200.0                          # critical angle not shown above this terrain (m)
KT_PER_MS = 1.943844


# ------------------------------------------------------------------ moisture
def es_hpa(t):
    """Saturation vapour pressure over water (Bolton 1980), t in K."""
    tc = t - 273.15
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def q_from_td(td, p):
    e = es_hpa(td)
    r = EPS * e / np.maximum(p - e, 1e-3)
    return r / (1 + r)


def qsat(t, p):
    return q_from_td(t, p)


def td_from_q(q, p):
    r = q / np.maximum(1 - q, 1e-6)
    e = np.maximum(p * r / (EPS + r), 1e-4)
    x = np.log(e / 6.112)
    return 243.5 * x / (17.67 - x) + 273.15


def theta(t, p):
    return t * (1000.0 / p) ** KAPPA


def t_lcl(t, td):
    """Bolton (1980) eq. 15."""
    return 1.0 / (1.0 / np.maximum(td - 56.0, 1.0) + np.log(t / td) / 800.0) + 56.0


def theta_e(t, p, q, tl=None):
    """Bolton (1980) eq. 43, pseudo-equivalent potential temperature."""
    r = 1000.0 * q / np.maximum(1 - q, 1e-6)          # g/kg
    if tl is None:
        tl = t_lcl(t, td_from_q(q, p))
    return (t * (1000.0 / p) ** (0.2854 * (1 - 0.28e-3 * r))
            * np.exp((3.376 / tl - 0.00254) * r * (1 + 0.81e-3 * r)))


# Pseudo-adiabats as a lookup table: T(theta_e, p) for saturated parcels.
_P0, _DP = 1100.0, 5.0
_E0, _DE = 200.0, 0.25
_TBL = None


def _table():
    global _TBL
    if _TBL is None:
        ps = np.arange(_P0, 49.0, -_DP)
        ts = np.arange(150.0, 340.0, 0.05)
        es_ax = np.arange(_E0, 480.0, _DE)
        tbl = np.empty((ps.size, es_ax.size), np.float32)
        for i, p in enumerate(ps):
            with np.errstate(over="ignore", invalid="ignore"):
                the = theta_e(ts, p, qsat(ts, p), tl=ts)
            the = np.nan_to_num(the, nan=1e4, posinf=1e4)
            the = np.maximum.accumulate(np.minimum(the, 1e4))   # monotone for interp
            tbl[i] = np.interp(es_ax, the, ts)
        _TBL = tbl
    return _TBL


def t_moist(the, p):
    """Temperature of a saturated parcel with pseudo-equivalent theta `the` at p."""
    tbl = _table()
    fi = (_P0 - p) / _DP
    fe = (the - _E0) / _DE
    shp = np.broadcast(the, p).shape
    fi = np.broadcast_to(fi, shp).ravel()
    fe = np.broadcast_to(fe, shp).ravel()
    return map_coordinates(tbl, [fi, fe], order=1, mode="nearest").reshape(shp)


# ------------------------------------------------------------------ helpers
def interp_at_p(f, p, ptarget):
    """Value of level field f (L,y,x) at pressure ptarget (y,x), linear in ln p.
    p must decrease with level (duplicates allowed)."""
    k = np.clip((p > ptarget[None]).sum(0) - 1, 0, p.shape[0] - 2)
    take = lambda a, kk: np.take_along_axis(a, kk[None], 0)[0]
    p0, p1 = take(p, k), take(p, k + 1)
    f0, f1 = take(f, k), take(f, k + 1)
    w = np.where(p0 > p1, np.log(p0 / ptarget) / np.log(np.maximum(p0 / np.maximum(p1, 1e-3), 1 + 1e-9)), 0.0)
    return f0 + np.clip(w, 0, 1) * (f1 - f0)


def interp_at_z(f, z, ztarget):
    """Value of level field f at height ztarget (scalar or (y,x)); z increases with level."""
    zt = np.broadcast_to(ztarget, z.shape[1:])
    k = np.clip((z <= zt[None]).sum(0) - 1, 0, z.shape[0] - 2)
    take = lambda a, kk: np.take_along_axis(a, kk[None], 0)[0]
    z0, z1 = take(z, k), take(z, k + 1)
    f0, f1 = take(f, k), take(f, k + 1)
    w = np.where(z1 > z0, (zt - z0) / np.maximum(z1 - z0, 1e-6), 0.0)
    return f0 + np.clip(w, 0, 1) * (f1 - f0)


# ------------------------------------------------------------------ column
def build_column(sfc, prof):
    """sfc: dict of (y,x): p (hPa), t, q, u, v, zs (m MSL), pbl (m), t_model, q_model, p_model.
    prof: dict of (L,y,x) on pressure levels (top-down order irrelevant; sorted here):
          p (hPa, level values broadcast), t, q, z (m MSL), u, v.
    Returns dict of (L+1,y,x) arrays, level 0 = surface, pressure decreasing."""
    order = np.argsort(-prof["p"][:, 0, 0])
    P = {k: prof[k][order].astype(np.float32) for k in ("p", "t", "q", "z", "u", "v")}
    zagl = P["z"] - sfc["zs"][None]

    # surface correction mixed through the model boundary layer
    pbl = np.maximum(sfc["pbl"], PBL_MIN_M)
    w = np.clip(1 - (zagl - pbl[None]) / (PBL_TAPER * pbl[None]), 0, 1)
    w = np.where(zagl < 0, 1.0, w)
    dth = theta(sfc["t"], sfc["p"]) - theta(sfc["t_model"], sfc["p_model"])
    dq = sfc["q"] - sfc["q_model"]
    th = theta(P["t"], P["p"]) + w * dth[None]
    P["t"] = th * (P["p"] / 1000.0) ** KAPPA
    P["q"] = np.clip(P["q"] + w * dq[None], 1e-6, qsat(P["t"], P["p"]))

    below = P["p"] >= sfc["p"][None] - 0.5
    col = {}
    for k, s in (("p", sfc["p"]), ("t", sfc["t"]), ("q", sfc["q"]), ("u", sfc["u"]),
                 ("v", sfc["v"]), ("z", sfc["zs"])):
        lev = np.where(below, s[None], P[k])
        col[k] = np.concatenate([s[None], lev]).astype(np.float32)
    col["zagl"] = np.maximum(col["z"] - sfc["zs"][None], 0)
    col["tv"] = col["t"] * (1 + 0.608 * col["q"])
    return col


# ------------------------------------------------------------------ parcels
def parcel(col, p0, t0, q0, k_start=None):
    """Lift a parcel from (p0, t0, q0). Layers below p0 don't count.
    Returns cape, cin (J/kg, cin <= 0, NaN where no LFC), p_lcl, z_lcl (m AGL)."""
    p, tv_e = col["p"], col["tv"]
    td0 = td_from_q(q0, p0)
    tl = t_lcl(t0, np.minimum(td0, t0))
    th0 = theta(t0, p0)
    p_l = 1000.0 * (tl / th0) ** (1 / KAPPA)
    the0 = theta_e(t0, p0, q0, tl)

    dry = p >= p_l[None]
    t_dry = th0[None] * (p / 1000.0) ** KAPPA
    t_wet = t_moist(np.broadcast_to(the0[None], p.shape), p)
    t_p = np.where(dry, t_dry, t_wet)
    q_p = np.where(dry, q0[None], qsat(t_wet, p))
    b = t_p * (1 + 0.608 * q_p) - tv_e                      # virtual temperature excess
    active = p <= p0[None] + 0.01
    b = np.where(active, b, 0.0)

    d = np.log(np.maximum(p[:-1], 1e-3) / np.maximum(p[1:], 1e-3))    # layer thickness in ln p
    d = np.where(active[:-1] & active[1:], d, 0.0)
    b0, b1 = b[:-1], b[1:]
    same = (b0 >= 0) == (b1 >= 0)
    tot = np.abs(b0) + np.abs(b1)
    tot = np.where(tot == 0, 1.0, tot)
    pos = np.where(same, np.where(b0 >= 0, 0.5 * (b0 + b1), 0.0),
                   np.maximum(b0, b1) ** 2 / (2 * tot)) * d
    neg = np.where(same, np.where(b0 < 0, 0.5 * (b0 + b1), 0.0),
                   -np.minimum(b0, b1) ** 2 / (2 * tot)) * d

    nl = pos.shape[0]
    kk = np.arange(nl)[:, None, None]
    above_lcl = p[1:] <= p_l[None]
    cand = (pos > 0) & above_lcl
    has = cand.any(0)
    lfc = np.argmax(cand, 0)
    el = nl - 1 - np.argmax((pos > 0)[::-1], 0)
    # Layers strictly between the LFC and EL layers count whole. In the LFC layer the
    # negative part lies below the crossing (CIN); in the EL layer it lies above the EL
    # (counts nowhere: at the tropopause that part alone can be -800 J/kg).
    at = lambda a, k: np.take_along_axis(a, k[None], 0)[0]
    inner = (kk > lfc[None]) & (kk < el[None])
    b0_lfc = at(b0, lfc)
    lfc_neg = at(neg, lfc)
    el_pos, el_neg, b1_el = at(pos, el), at(neg, el), at(b1, el)
    cape = (np.where(inner, pos + neg, 0.0).sum(0) + at(pos, lfc)
            + np.where(b0_lfc < 0, 0.0, lfc_neg)
            + np.where(el > lfc, el_pos + np.where(b1_el < 0, 0.0, el_neg), 0.0))
    cape = RD * cape
    cin = RD * (np.where(kk < lfc[None], neg, 0.0).sum(0) + np.where(b0_lfc < 0, lfc_neg, 0.0))
    cape = np.where(has, np.maximum(cape, 0), 0.0)
    cin = np.where(has, cin, np.nan)
    z_l = interp_at_p(col["zagl"], p, p_l)
    return cape.astype(np.float32), cin.astype(np.float32), p_l.astype(np.float32), z_l.astype(np.float32)


def ml_parcel_start(col, depth=ML_DEPTH_HPA):
    """Mean theta and q of the lowest `depth` hPa (pressure weighted)."""
    p = col["p"]
    ps = p[0]
    top = ps - depth
    th = theta(col["t"], p)
    q = col["q"]
    hi = np.minimum(p[:-1], ps[None])
    lo = np.maximum(p[1:], top[None])
    wgt = np.clip(hi - lo, 0, None)
    th_m = (wgt * 0.5 * (th[:-1] + th[1:])).sum(0) / np.maximum(wgt.sum(0), 1e-3)
    q_m = (wgt * 0.5 * (q[:-1] + q[1:])).sum(0) / np.maximum(wgt.sum(0), 1e-3)
    return ps, th_m * (ps / 1000.0) ** KAPPA, q_m


def mu_parcel_start(col, depth=MU_DEPTH_HPA):
    p = col["p"]
    the = theta_e(col["t"], p, col["q"])
    the = np.where(p >= p[0][None] - depth, the, -np.inf)
    k = np.argmax(the, 0)
    take = lambda a: np.take_along_axis(a, k[None], 0)[0]
    return take(p), take(col["t"]), take(col["q"])


# ------------------------------------------------------------------ winds
HGT_STEP = 100.0


def wind_profile(col, top=6000.0):
    hs = np.arange(0.0, top + 1, HGT_STEP)
    u = np.stack([interp_at_z(col["u"], col["zagl"], h) for h in hs])
    v = np.stack([interp_at_z(col["v"], col["zagl"], h) for h in hs])
    return hs, u, v


def bunkers_right(hs, u, v):
    """Bunkers et al. (2000) right mover, non-pressure-weighted 0-6 km mean wind."""
    um, vm = u[hs <= 6000].mean(0), v[hs <= 6000].mean(0)
    ub, vb = u[hs <= 500].mean(0), v[hs <= 500].mean(0)
    ut, vt = u[(hs >= 5500) & (hs <= 6000)].mean(0), v[(hs >= 5500) & (hs <= 6000)].mean(0)
    su, sv = ut - ub, vt - vb
    mag = np.maximum(np.hypot(su, sv), 1e-6)
    return um + 7.5 * sv / mag, vm - 7.5 * su / mag


def srh(hs, u, v, cu, cv, depth):
    k = hs <= depth + 1e-6
    su, sv = u[k] - cu[None], v[k] - cv[None]
    return (su[1:] * sv[:-1] - su[:-1] * sv[1:]).sum(0)


def bulk_shear(hs, u, v, depth):
    k = int(round(depth / HGT_STEP))
    return u[k] - u[0], v[k] - v[0]


# ------------------------------------------------------------------ composites
def stp_fixed(sbcape, sblcl, srh1, bwd6):
    """SPC fixed-layer significant tornado parameter (Thompson et al. 2003/2012)."""
    lcl = np.clip((2000 - sblcl) / 1000, 0, 1)
    shr = np.where(bwd6 < 12.5, 0, np.minimum(bwd6, 30) / 20)
    return np.maximum(sbcape / 1500 * lcl * np.maximum(srh1, 0) / 150 * shr, 0)


def scp_fixed(mucape, srh3, bwd6):
    """Supercell composite with fixed layers (SPC's map uses the effective layer)."""
    shr = np.where(bwd6 < 10, 0, np.minimum(bwd6, 20) / 20)
    return np.maximum(mucape / 1000 * np.maximum(srh3, 0) / 50 * shr, 0)


def mfc(q, u, v, dx, smooth=1.5):
    """Surface moisture flux convergence, g/kg per hour (positive = convergence)."""
    qu, qv = gaussian_filter(q * u, smooth), gaussian_filter(q * v, smooth)
    div = np.gradient(qu, dx, axis=1) + np.gradient(qv, dx, axis=0)
    return -div * 1000 * 3600


def mslp(psfc, t2, zs):
    """Surface pressure reduced to sea level (standard-lapse approximation), hPa."""
    return psfc * (1 - 0.0065 * zs / (t2 + 0.0065 * zs)) ** -5.257


# ------------------------------------------------------------------ frame
def analyse(sfc, prof, dx):
    col = build_column(sfc, prof)
    out = {}
    sb = parcel(col, col["p"][0], col["t"][0], col["q"][0])
    out["sbcape"], out["sbcin"], _, out["sblcl"] = sb
    ml = parcel(col, *ml_parcel_start(col))
    out["mlcape"], out["mlcin"], _, out["mllcl"] = ml
    mu = parcel(col, *mu_parcel_start(col))
    out["mucape"] = mu[0]

    hs, u, v = wind_profile(col)
    cu, cv = bunkers_right(hs, u, v)
    out["rm_u"], out["rm_v"] = cu, cv
    out["srh1"] = srh(hs, u, v, cu, cv, 1000)
    out["srh3"] = srh(hs, u, v, cu, cv, 3000)
    out["shr6_u"], out["shr6_v"] = bulk_shear(hs, u, v, 6000)
    out["shr1_u"], out["shr1_v"] = bulk_shear(hs, u, v, 1000)
    bwd6 = np.hypot(out["shr6_u"], out["shr6_v"])
    out["bwd6"] = bwd6
    out["stp"] = stp_fixed(out["sbcape"], out["sblcl"], out["srh1"], bwd6)
    out["scp"] = scp_fixed(out["mucape"], out["srh3"], bwd6)

    out["t2"], out["td2"] = sfc["t"], td_from_q(sfc["q"], sfc["p"])
    out["u10"], out["v10"] = sfc["u"], sfc["v"]
    out["psfc"] = sfc["p"]
    out["zs"] = sfc["zs"]
    out["mslp"] = gaussian_filter(mslp(sfc["p"], sfc["t"], sfc["zs"]), 2)
    out["mfc"] = mfc(sfc["q"], sfc["u"], sfc["v"], dx)
    t5 = interp_at_p(col["t"], col["p"], np.full_like(sfc["p"], 500.0))
    t7 = interp_at_p(col["t"], col["p"], np.full_like(sfc["p"], 700.0))
    z5 = interp_at_p(col["z"], col["p"], np.full_like(sfc["p"], 500.0))
    z7 = interp_at_p(col["z"], col["p"], np.full_like(sfc["p"], 700.0))
    out["lr75"] = (t7 - t5) / np.maximum(z5 - z7, 1) * 1000          # 700-500 hPa lapse rate, C/km
    out.update(visibility(col, hs, u, v, cu, cv, sfc["zs"]))
    return {k: np.asarray(a, np.float32) for k, a in out.items()}


def precipitable_water(col):
    """kg/m2 (= mm) from the surface to the top level (100 hPa). Below-ground levels
    are zero-thickness copies of the surface, so they add nothing."""
    p, q = col["p"] * 100.0, col["q"]
    return (0.5 * (q[1:] + q[:-1]) * (p[:-1] - p[1:])).sum(0) / G


def visibility(col, hs, u, v, cu, cv, zs):
    """Tornado Visibility's mode index (0 = HP, 0.5 = Classic, 1 = LP) and critical angle,
    with MesoXV's inputs: the RTMA 10 m wind, MesoXV's Bunkers motion (SPC convention, not
    HRRR's own storm motion), HRRR's 300 mb wind and the column's precipitable water.
    The STP mask (>= 0.5) is applied when drawing."""
    sfcp = col["p"][0]
    u3 = interp_at_p(col["u"], col["p"], np.full_like(sfcp, 300.0))
    v3 = interp_at_p(col["v"], col["p"], np.full_like(sfcp, 300.0))
    sr300 = np.hypot(u3 - cu, v3 - cv) * KT_PER_MS
    sr_u10, sr_v10 = u[0] - cu, v[0] - cv
    llsr = np.hypot(sr_u10, sr_v10) * KT_PER_MS
    pw_in = precipitable_water(col) / 25.4
    sr_term = np.clip((sr300 - TV_SR_FLOOR_KT) / (TV_SR_CEIL_KT - TV_SR_FLOOR_KT), 0, 1)
    ll_term = np.clip((llsr - TV_LLSR_FLOOR_KT) / (TV_LLSR_CEIL_KT - TV_LLSR_FLOOR_KT), 0, 1)
    pw_term = (np.clip((TV_PW_NEUTRAL_IN - pw_in) / TV_PW_DRY_IN, 0, 1)
               + np.clip((TV_PW_NEUTRAL_IN - pw_in) / TV_PW_WET_IN, -1, 0)) * TV_PW_WEIGHT
    tvis = np.clip(np.maximum(sr_term, ll_term) + pw_term, 0, 1)
    # critical angle: 10 m storm-relative wind vs the 0-500 m shear vector (Esterheld &
    # Giuliano 2008). MesoXV has the 500 m wind from its 100 m profile, so no 80 m / 925 mb
    # interpolation is needed as in Tornado Visibility.
    k5 = int(round(500.0 / HGT_STEP))
    sh_u, sh_v = u[k5] - u[0], v[k5] - v[0]
    m_sr, m_sh = np.hypot(sr_u10, sr_v10), np.hypot(sh_u, sh_v)
    cosa = np.clip((sr_u10 * sh_u + sr_v10 * sh_v) / np.maximum(m_sr * m_sh, 1e-9), -1, 1)
    crit = np.degrees(np.arccos(cosa))
    crit = np.where((m_sr > 2.0) & (m_sh > 1.0) & (zs < TV_CA_OROG_MAX), crit, np.nan)
    return {"tvis": tvis, "sr300": sr300, "llsr": llsr, "pw": pw_in, "crit": crit}
