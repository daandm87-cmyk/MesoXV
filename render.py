"""MesoXV maps: one PNG per product, region and frame.

US chasing units like SPC's pages: F, knots, J/kg, m2/s2, mb.
Palettes are colour-vision-deficiency safe: every one here is single-hue sequential
or purple-orange diverging, and CIN is shown with blue hatching, never with
a green/red contrast.
"""
from __future__ import annotations

import io

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

HRRR_CRS = ccrs.LambertConformal(
    central_longitude=-97.5, central_latitude=38.5, standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(ellipse="sphere", semimajor_axis=6371229, semiminor_axis=6371229))
PC = ccrs.PlateCarree()

plt.rcParams["hatch.color"] = "#0072B2"
plt.rcParams["hatch.linewidth"] = 0.6

KT = 1.943844
BLUE = "#0072B2"       # Okabe-Ito
PURPLE = "#8E4FA0"
INK = "#1d2a35"

REGIONS = {
    "plains":     ("Plains", [-106.0, -86.0, 29.0, 47.0]),
    "midwest":    ("Midwest", [-100.0, -84.0, 37.0, 46.5]),
    "iowa":       ("Iowa", [-98.0, -89.0, 39.8, 44.6]),
    "s_plains":   ("Southern Plains", [-104.5, -92.0, 29.5, 38.5]),
    "c_plains":   ("Central Plains", [-104.5, -92.5, 35.5, 43.5]),
    "n_plains":   ("Northern Plains", [-105.0, -93.0, 41.0, 49.0]),
    "ohio_valley": ("Ohio Valley", [-93.0, -80.5, 35.0, 42.0]),
    "southeast":  ("Southeast", [-95.0, -79.0, 29.0, 37.5]),
}


def f_of(k):
    return (k - 273.15) * 9 / 5 + 32


# ------------------------------------------------------------------ products
# Each product: title, units, fill (field fn, levels, cmap, extend), overlays.
def _cin_hatch(key):
    return ("cin", key)


RADAR = ("radar",)        # MRMS composite reflectivity outlines, 35 and 50 dBZ


PRODUCTS = {
    "sfc": dict(title="Surface: dewpoint, pressure, wind", units="Dewpoint (F)",
                fill=(lambda d: f_of(d["td2"]), [40, 45, 50, 55, 58, 60, 62, 64, 66, 68, 70, 72, 75], "YlGnBu", "both"),
                over=[("isobars",), ("temp",), ("barbs", "u10", "v10")]),
    "mfc": dict(title="Moisture convergence + surface wind", units="Moisture flux convergence (g/kg per hour)",
                fill=(lambda d: d["mfc"], [0.5, 1, 2, 3, 4, 6, 8, 10], "Purples", "max"),
                over=[("td_line", 60), ("barbs", "u10", "v10"), RADAR],
                show=(lambda d: np.where(f_of(d["td2"]) >= 50, d["mfc"], np.nan), 2.0,
                      "no moisture convergence of 2+ g/kg per hour with dewpoints of 50F or more")),
    "sbcape": dict(title="Surface-based CAPE + CIN", units="SBCAPE (J/kg)  ·  dashed blue: SBCIN -25, hatched: worse than -100",
                   fill=(lambda d: d["sbcape"], [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000], "YlOrBr", "max"),
                   over=[_cin_hatch("sbcin"), RADAR],
                   show=(lambda d: d["sbcape"], 250, "no SBCAPE of 250 J/kg or more")),
    "mlcape": dict(title="Mixed-layer CAPE + CIN (100 mb)", units="MLCAPE (J/kg)  ·  dashed blue: MLCIN -25, hatched: worse than -100",
                   fill=(lambda d: d["mlcape"], [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000], "YlOrBr", "max"),
                   over=[_cin_hatch("mlcin"), RADAR],
                   show=(lambda d: d["mlcape"], 250, "no MLCAPE of 250 J/kg or more")),
    "mucape": dict(title="Most-unstable CAPE", units="MUCAPE (J/kg)",
                   fill=(lambda d: d["mucape"], [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000], "YlOrBr", "max"),
                   over=[RADAR],
                   show=(lambda d: d["mucape"], 250, "no MUCAPE of 250 J/kg or more")),
    "lcl": dict(title="Mixed-layer LCL height", units="MLLCL (m above ground)  ·  lower favours tornadoes",
                fill=(lambda d: np.where(d["mlcape"] > 50, d["mllcl"], np.nan), [500, 750, 1000, 1250, 1500, 2000, 2500],
                      "PuBu_r", "both"),
                over=[],
                show=(lambda d: d["mlcape"], 250, "no MLCAPE of 250 J/kg or more, so LCL heights don't matter here")),
    "srh1": dict(title="0-1 km storm-relative helicity + Bunkers right mover", units="SRH 0-1 km (m2/s2)",
                 fill=(lambda d: d["srh1"], [50, 100, 150, 200, 250, 300, 400, 500, 600], "RdPu", "max"),
                 over=[("arrows", "rm_u", "rm_v"), RADAR],
                 show=(lambda d: np.where(d["mlcape"] >= 100, d["srh1"], np.nan), 100,
                       "no 0-1 km SRH of 100+ where there's instability (MLCAPE 100+)")),
    "srh1obs": dict(title="0-1 km SRH: radar-observed vs MesoXV",
                    units="Circles at radars: observed minus MesoXV 0-1 km SRH (m2/s2), number = observed  ·  orange = more than MesoXV has",
                    fill=(lambda d: d["srh1"], [50, 100, 150, 200, 250, 300, 400, 500, 600], "RdPu", "max"),
                    over=[("vad",), RADAR], show="vad"),
    "srh3": dict(title="0-3 km storm-relative helicity + Bunkers right mover", units="SRH 0-3 km (m2/s2)",
                 fill=(lambda d: d["srh3"], [100, 150, 200, 250, 300, 400, 500, 600, 800], "RdPu", "max"),
                 over=[("arrows", "rm_u", "rm_v"), RADAR],
                 show=(lambda d: np.where(d["mucape"] >= 100, d["srh3"], np.nan), 150,
                       "no 0-3 km SRH of 150+ where there's instability (MUCAPE 100+)")),
    "shear": dict(title="0-6 km bulk shear", units="0-6 km shear (kt)",
                  fill=(lambda d: d["bwd6"] * KT, [30, 35, 40, 45, 50, 60, 70, 80], "Blues", "max"),
                  over=[("barbs", "shr6_u", "shr6_v"), RADAR],
                  show=(lambda d: np.where(d["mucape"] >= 100, d["bwd6"] * KT, np.nan), 35,
                        "no 0-6 km shear of 35+ kt where there's instability (MUCAPE 100+)")),
    "stp": dict(title="Significant tornado parameter (fixed layer)", units="STP  ·  dashed blue: MLCIN -25, hatched: worse than -100",
                fill=(lambda d: d["stp"], [0.5, 1, 2, 3, 4, 6, 8, 10], "magma_r", "max"),
                over=[_cin_hatch("mlcin"), RADAR],
                show=(lambda d: d["stp"], 0.5, "no STP of 0.5 or more")),
    "scp": dict(title="Supercell composite (fixed layer)", units="SCP",
                fill=(lambda d: d["scp"], [1, 2, 4, 6, 8, 12, 16, 20], "BuPu", "max"),
                over=[_cin_hatch("mlcin"), RADAR],
                show=(lambda d: d["scp"], 1, "no SCP of 1 or more")),
    "lr75": dict(title="700-500 mb lapse rate", units="Lapse rate (C/km)",
                 fill=(lambda d: d["lr75"], [6, 6.5, 7, 7.5, 8, 8.5, 9], "Oranges", "max"),
                 over=[],
                 show=(lambda d: d["lr75"], 7.0, "no 700-500 mb lapse rates of 7 C/km or more")),
    "pchg": dict(title="1-hour pressure change", units="Sea-level pressure change over the last hour (mb)  ·  purple = falling",
                 fill=(lambda d: d.get("dmslp1h"), [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4], "PuOr_r", "both"),
                 over=[("isobars",), RADAR], needs="dmslp1h",
                 show=(lambda d: np.abs(d["dmslp1h"]), 1.0, "no pressure changes of 1 mb or more in the last hour")),
    "capechg": dict(title="1-hour MLCAPE change", units="MLCAPE change over the last hour (J/kg)  ·  orange = rising",
                    fill=(lambda d: d.get("dmlcape1h"), [-1500, -1000, -500, -250, -100, 100, 250, 500, 1000, 1500], "PuOr_r", "both"),
                    over=[_cin_hatch("mlcin"), RADAR], needs="dmlcape1h",
                    show=(lambda d: np.abs(d["dmlcape1h"]), 250, "no MLCAPE changes of 250+ J/kg in the last hour")),
    "cinchg": dict(title="1-hour MLCIN change (cap weakening)", units="MLCIN change over the last hour (J/kg), where unstable then and now  ·  orange = cap weakening",
                   fill=(lambda d: d.get("dmlcin1h"), [-100, -50, -25, -10, 10, 25, 50, 100], "PuOr_r", "both"),
                   over=[("cin_line", "mlcin"), RADAR], needs="dmlcin1h",
                   show=(lambda d: np.abs(d["dmlcin1h"]), 25, "no MLCIN changes of 25+ J/kg in unstable air in the last hour")),
}
ORDER = ["sfc", "mfc", "pchg", "sbcape", "mlcape", "mucape", "capechg", "cinchg", "lcl", "lr75",
         "srh1", "srh1obs", "srh3", "shear", "stp", "scp"]

_FEATURES = {}


def _region_features(key, extent):
    if key not in _FEATURES:
        w, e, s, n = extent
        bb = box(w - 1, s - 1, e + 1, n + 1)
        counties = [g for g in cfeature.NaturalEarthFeature("cultural", "admin_2_counties", "10m").geometries()
                    if g.intersects(bb)]
        _FEATURES[key] = counties
    return _FEATURES[key]


def region_mask(grid, region):
    w, e, s, n = REGIONS[region][1]
    return (grid.lon >= w) & (grid.lon <= e) & (grid.lat >= s) & (grid.lat <= n)


def worth_showing(d, grid, product, region):
    """(True, "") or (False, why): skip maps with nothing worth seeing in the region.
    The thresholds live in each product's `show` rule above."""
    spec = PRODUCTS[product]
    if spec.get("needs") and spec["needs"] not in d:
        return False, "This map needs the frame from an hour earlier."
    rule = spec.get("show")
    if rule is None:
        return True, ""
    name = REGIONS[region][0]
    if rule == "vad":
        w, e, s, n = REGIONS[region][1]
        pts = [o for o in d.get("_vad", []) if w <= o["lon"] <= e and s <= o["lat"] <= n]
        if not pts:
            return False, "No radar wind profiles reaching 1 km in %s at this time." % name
        if max(max(o["obs"], o["ana"]) for o in pts) < 100:
            return False, "Nothing to show in %s: no radar measures (or MesoXV has) 0-1 km SRH of 100+." % name
        return True, ""
    fn, thr, why = rule
    vals = np.asarray(fn(d))[region_mask(grid, region)]
    if np.isfinite(vals).any() and np.nanmax(vals) >= thr:
        return True, ""
    return False, "Nothing to show in %s: %s." % (name, why)


def _thin_step(shape, target=26):
    return max(1, int(round(max(shape) / target)))


def render(d, grid, product, region, label, source):
    """PNG bytes for one product/region. d = frame dict, grid = AnalysisGrid."""
    spec = PRODUCTS[product]
    name, extent = REGIONS[region]
    x, y = grid.x, grid.y
    # figure shaped like the region in the map projection
    pts = HRRR_CRS.transform_points(PC, np.array([extent[0], extent[1], extent[0], extent[1]]),
                                    np.array([extent[2], extent[2], extent[3], extent[3]]))
    aspect = (pts[:, 1].max() - pts[:, 1].min()) / (pts[:, 0].max() - pts[:, 0].min())
    w_in = 11.0
    h_in = w_in * aspect + 1.55            # map + 0.25 in title strip + 1.3 in colour bar / credits
    fig = plt.figure(figsize=(w_in, h_in), dpi=110)
    ax = fig.add_axes([0.01, 1.3 / h_in, 0.98, (w_in * aspect) / h_in], projection=HRRR_CRS)
    ax.set_extent(extent, crs=PC)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f2ee", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfe7ee", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#dfe7ee", edgecolor="#9fb0bf", lw=0.4, zorder=0)

    fn, levels, cmap, extend = spec["fill"]
    field = fn(d)
    cs = None
    if field is not None:
        cm = plt.get_cmap(cmap)
        norm = BoundaryNorm(levels, cm.N, extend=extend)
        cs = ax.contourf(x, y, field, levels=levels, cmap=cm, norm=norm, extend=extend,
                         transform=HRRR_CRS, alpha=0.88, zorder=1)
    else:
        ax.text(0.5, 0.5, "Needs an hour of earlier frames", transform=ax.transAxes,
                ha="center", va="center", fontsize=15, color=INK)

    st = _thin_step(field.shape if field is not None else d["u10"].shape)
    cs_override = None
    for ov in spec["over"]:
        kind = ov[0]
        if kind == "cin":
            cin = np.nan_to_num(d[ov[1]], nan=0.0)
            ax.contourf(x, y, cin, levels=[-1e4, -100], colors="none", hatches=["////"],
                        transform=HRRR_CRS, zorder=2)
            ax.contour(x, y, cin, levels=[-100, -25], colors=BLUE, linewidths=[1.1, 0.7],
                       linestyles=["solid", "dashed"], transform=HRRR_CRS, zorder=2)
        elif kind == "cin_line":
            cin = np.nan_to_num(d[ov[1]], nan=0.0)
            ax.contour(x, y, cin, levels=[-100, -50, -25], colors=BLUE, linewidths=[1.2, 0.9, 0.7],
                       transform=HRRR_CRS, zorder=2)
        elif kind == "isobars":
            p = d["mslp"]
            lv = np.arange(np.floor(np.nanmin(p) / 2) * 2, np.nanmax(p) + 2, 2)
            c = ax.contour(x, y, p, levels=lv, colors=INK, linewidths=0.9, transform=HRRR_CRS, zorder=3)
            ax.clabel(c, fmt="%d", fontsize=8, inline=True)
        elif kind == "temp":
            t = f_of(d["t2"])
            lv = np.arange(np.floor(np.nanmin(t) / 5) * 5, np.nanmax(t) + 5, 5)
            c = ax.contour(x, y, t, levels=lv, colors="#b0563a", linewidths=0.6, linestyles="dashed",
                           transform=HRRR_CRS, zorder=3)
            ax.clabel(c, fmt="%d", fontsize=7, inline=True)
        elif kind == "td_line":
            c = ax.contour(x, y, f_of(d["td2"]), levels=[ov[1]], colors="#2a7f62", linewidths=1.3,
                           transform=HRRR_CRS, zorder=3)
            ax.clabel(c, fmt="%dF Td", fontsize=8, inline=True)
        elif kind == "barbs":
            u, v = d[ov[1]] * KT, d[ov[2]] * KT
            ax.barbs(x[::st], y[::st], u[::st, ::st], v[::st, ::st], length=5.2, linewidth=0.55,
                     color=INK, transform=HRRR_CRS, zorder=4)
        elif kind == "radar":
            if d.get("refl") is not None:
                z = gaussian_filter(d["refl"], 0.6)
                ax.contour(x, y, z, levels=[35, 50], colors=["#5a5a5a", "#111111"], linewidths=[0.6, 1.1],
                           transform=HRRR_CRS, zorder=4)
        elif kind == "vad":
            pts = d.get("_vad", [])
            if pts:
                vlev = [-150, -100, -50, -25, 25, 50, 100, 150]
                vcm = plt.get_cmap("PuOr_r")
                vnorm = BoundaryNorm(vlev, vcm.N, extend="both")
                vs = ax.scatter([o["x"] for o in pts], [o["y"] for o in pts], c=[o["diff"] for o in pts],
                                cmap=vcm, norm=vnorm, s=190, edgecolors=INK, linewidths=0.9,
                                transform=HRRR_CRS, zorder=6)
                for o in pts:
                    ax.text(o["x"], o["y"], "  %d" % o["obs"], transform=HRRR_CRS, fontsize=8.5,
                            fontweight="bold", color=INK, ha="left", va="center", zorder=7, clip_on=True,
                            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
                cs_override = (vs, vlev)
                if cs is not None:
                    cs.set_alpha(0.4)               # SRH shading as background only
        elif kind == "arrows":
            u, v = d[ov[1]] * KT, d[ov[2]] * KT
            q = ax.quiver(x[::st], y[::st], u[::st, ::st], v[::st, ::st], color=INK, scale=900, width=0.0022,
                          transform=HRRR_CRS, zorder=4)
            ax.quiverkey(q, 0.97, -0.035, 40, "40 kt storm motion", labelpos="W", fontproperties={"size": 8})

    ax.add_geometries(_region_features(region, extent), PC, facecolor="none", edgecolor="#9a9a9a",
                      linewidth=0.25, zorder=5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#3b3b3b", linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#3b3b3b", linewidth=0.9, zorder=5)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#3b3b3b", linewidth=0.7, zorder=5)

    fig.text(0.012, 0.995, "MesoXV  ·  %s" % spec["title"], ha="left", va="top", fontsize=12.5,
             fontweight="bold", color=INK)
    fig.text(0.988, 0.995, label, ha="right", va="top", fontsize=11.5, fontweight="bold", color=INK)
    if cs_override is not None:                 # the radar map's colour bar is the difference scale
        cs, levels = cs_override
    if cs is not None:
        cax = fig.add_axes([0.2, 0.78 / h_in, 0.6, 0.22 / h_in])
        cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=levels)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(spec["units"], fontsize=9)
    fig.text(0.012, 0.06 / h_in, source, ha="left", va="bottom", fontsize=7.5, color="#555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    # 256-colour palette PNG: about a third of the size, no visible difference on these maps
    from PIL import Image
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
