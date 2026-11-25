"""Load the inputs for one 15-minute frame and run the analysis."""
from __future__ import annotations

import datetime as dt

import numpy as np

import data
import grids
import analysis as an

PRS_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 800, 750, 700, 650, 600, 550, 500,
              450, 400, 350, 300, 250, 200, 150, 100]
WIND_TOP_HPA = 300           # winds above this aren't needed (0-6 km AGL)

HRRR_SFC = [("PRES", "surface"), ("HGT", "surface"), ("TMP", "2 m above ground"),
            ("SPFH", "2 m above ground"), ("UGRD", "10 m above ground"),
            ("VGRD", "10 m above ground"), ("HPBL", "surface")]
HRRR_CHECK = [("CAPE", "surface"), ("CIN", "surface"), ("CAPE", "90-0 mb above ground"),
              ("CIN", "90-0 mb above ground"), ("CAPE", "255-0 mb above ground"),
              ("HLCY", "1000-0 m above ground"), ("HLCY", "3000-0 m above ground")]
RTMA = [("PRES", "surface"), ("HGT", "surface"), ("TMP", "2 m above ground"),
        ("DPT", "2 m above ground"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")]

RTMA_DELAY = dt.timedelta(minutes=18)    # RTMA-RU lands ~15-18 min after valid time
HRRR_DELAY = dt.timedelta(minutes=55)    # an HRRR run's first hours are up ~50-55 min after init


def floor_hour(t):
    return t.replace(minute=0, second=0, microsecond=0)


def hrrr_run_for(valid):
    """The newest HRRR run you'd actually have when this frame's RTMA arrives."""
    return floor_hour(valid + RTMA_DELAY - HRRR_DELAY)


def prs_wanted():
    w = []
    for p in PRS_LEVELS:
        lev = "%d mb" % p
        w += [("TMP", lev), ("SPFH", lev), ("HGT", lev)]
        if p >= WIND_TOP_HPA:
            w += [("UGRD", lev), ("VGRD", lev)]
    return w


class Inputs:
    """Caches HRRR hours (already cut to the analysis grid) between frames."""

    def __init__(self, extent, step=2, check=False):
        self.extent, self.step, self.check = extent, step, check
        self.grid = None
        self._hrrr = {}
        self._rtma_remap = None
        self._rtma_grid = None
        self._rtma_zs = None

    def _ensure_grid(self, hgrid):
        if self.grid is None:
            self.grid = grids.AnalysisGrid(hgrid, self.extent, self.step)

    def hrrr(self, run, fxx):
        key = (run, fxx)
        if key in self._hrrr:
            return self._hrrr[key]
        sfc_w = HRRR_SFC + (HRRR_CHECK if self.check else [])
        sf, hgrid = data.load(data.hrrr_url(run, fxx, "sfc"), sfc_w)
        self._ensure_grid(hgrid)
        g = self.grid
        pr, _ = data.load(data.hrrr_url(run, fxx, "prs"), prs_wanted())
        nl = len(PRS_LEVELS)
        prof = {k: np.full((nl,) + g.shape, np.nan, np.float32) for k in ("t", "q", "z", "u", "v")}
        for i, p in enumerate(PRS_LEVELS):
            lev = "%d mb" % p
            prof["t"][i] = g.take(pr[("TMP", lev)])
            prof["q"][i] = g.take(pr[("SPFH", lev)])
            prof["z"][i] = g.take(pr[("HGT", lev)])
            if p >= WIND_TOP_HPA:
                prof["u"][i] = g.take(pr[("UGRD", lev)])
                prof["v"][i] = g.take(pr[("VGRD", lev)])
            else:                                   # not used below 6 km AGL; keep finite
                prof["u"][i] = prof["u"][i - 1]
                prof["v"][i] = prof["v"][i - 1]
        prof["p"] = np.broadcast_to(np.array(PRS_LEVELS, np.float32)[:, None, None],
                                    prof["t"].shape).copy()
        sfc = {"p": g.take(sf[("PRES", "surface")]) / 100.0,
               "zs": g.take(sf[("HGT", "surface")]),
               "t": g.take(sf[("TMP", "2 m above ground")]),
               "q": g.take(sf[("SPFH", "2 m above ground")]),
               "u": g.take(sf[("UGRD", "10 m above ground")]),
               "v": g.take(sf[("VGRD", "10 m above ground")]),
               "pbl": g.take(sf[("HPBL", "surface")])}
        check = {("%s %s" % w): g.take(sf[w]) for w in HRRR_CHECK} if self.check else {}
        self._hrrr[key] = (sfc, prof, check)
        if len(self._hrrr) > 6:
            self._hrrr.pop(next(iter(self._hrrr)))
        return self._hrrr[key]

    def hrrr_at(self, valid):
        """HRRR surface + profile at `valid`, linear in time between two hours."""
        h0 = floor_hour(valid)
        a = valid.minute / 60.0
        run = hrrr_run_for(valid)
        for attempt in range(3):          # newest run not fully posted yet: use the one before
            f0 = int((h0 - run).total_seconds() // 3600)
            try:
                s0, p0, c0 = self.hrrr(run, f0)
                if a == 0:
                    return run, s0, p0, c0
                s1, p1, c1 = self.hrrr(run, f0 + 1)
                break
            except data.Missing:
                if attempt == 2:
                    raise
                run -= dt.timedelta(hours=1)
        mix = lambda x, y: (1 - a) * x + a * y
        sfc = {k: mix(s0[k], s1[k]) for k in s0}
        prof = {k: mix(p0[k], p1[k]) for k in p0}
        check = {k: mix(c0[k], c1[k]) for k in c0}
        return run, sfc, prof, check

    def rtma(self, valid):
        f, rgrid = data.load(data.rtma_url(valid), RTMA)
        g = self.grid
        if self._rtma_remap is None:
            self._rtma_remap = g.remapper(rgrid)
            self._rtma_grid = rgrid
        r = self._rtma_remap
        out = {"p": r(f[("PRES", "surface")]) / 100.0, "zs": r(f[("HGT", "surface")]),
               "t": r(f[("TMP", "2 m above ground")]), "td": r(f[("DPT", "2 m above ground")])}
        ue, ve = grids.grid_to_earth(f[("UGRD", "10 m above ground")], f[("VGRD", "10 m above ground")],
                                     rgrid, self._rtma_lon(rgrid))
        ue, ve = r(ue), r(ve)
        out["u"], out["v"] = grids.earth_to_grid(ue, ve, g.hgrid, g.lon)
        return out

    def _rtma_lon(self, rgrid):
        if getattr(self, "_rlon", None) is None:
            p = grids.proj_of(rgrid)
            x0, y0 = grids.origin_xy(rgrid)
            xs = x0 + np.arange(rgrid["nx"]) * rgrid["DxInMetres"]
            ys = y0 + np.arange(rgrid["ny"]) * rgrid["DyInMetres"]
            X, Y = np.meshgrid(xs, ys)
            self._rlon, _ = p(X, Y, inverse=True)
        return self._rlon


def make_frame(inp: Inputs, valid: dt.datetime, model_only=False):
    """Analysis for one valid time. model_only=True uses HRRR's own surface
    (for checking the parcel/wind code against HRRR's own CAPE and SRH)."""
    run, hs, prof, check = inp.hrrr_at(valid)
    g = inp.grid
    if model_only:
        s = dict(hs)
    else:
        r = inp.rtma(valid)
        # RTMA pressure is at RTMA's terrain height; move it to HRRR's (hydrostatic)
        tv = r["t"] * (1 + 0.608 * an.q_from_td(r["td"], r["p"]))
        p_adj = r["p"] * np.exp(-G_OVER_RD * (hs["zs"] - r["zs"]) / tv)
        s = {"p": p_adj, "t": r["t"], "q": an.q_from_td(np.minimum(r["td"], r["t"]), p_adj),
             "u": r["u"], "v": r["v"], "zs": hs["zs"], "pbl": hs["pbl"]}
        for k in ("p", "t", "q", "u", "v"):                      # RTMA gaps (offshore): use HRRR
            bad = ~np.isfinite(s[k])
            if bad.any():
                s[k] = np.where(bad, hs[k], s[k])
    s["t_model"], s["q_model"], s["p_model"] = hs["t"], hs["q"], hs["p"]
    s["zs"], s["pbl"] = hs["zs"], hs["pbl"]
    out = an.analyse(s, prof, g.dx)
    out["_run"] = run
    out["_check"] = check
    return out


G_OVER_RD = an.G / an.RD
