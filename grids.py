"""Analysis grid (a cropped, thinned piece of the HRRR grid), remapping of
the RTMA 2.5 km grid onto it, and wind rotation between the two Lambert grids.

Everything is computed and drawn in HRRR's own Lambert projection, so HRRR
grid-relative winds are already right for the maps; RTMA winds are rotated
into that frame.
"""
from __future__ import annotations

import numpy as np
import pyproj
from scipy.ndimage import map_coordinates


def proj_of(grid):
    return pyproj.Proj(proj="lcc", lat_1=grid["Latin1InDegrees"], lat_2=grid["Latin2InDegrees"],
                       lat_0=grid["Latin1InDegrees"], lon_0=grid["LoVInDegrees"] - 360.0,
                       R=grid["radius"])


def origin_xy(grid):
    p = proj_of(grid)
    lon0 = grid["longitudeOfFirstGridPointInDegrees"]
    lon0 = lon0 - 360.0 if lon0 > 180 else lon0
    return p(lon0, grid["latitudeOfFirstGridPointInDegrees"])


def cone(grid):
    """Cone constant n of a Lambert grid (secant or tangent)."""
    l1, l2 = np.radians(grid["Latin1InDegrees"]), np.radians(grid["Latin2InDegrees"])
    if abs(l1 - l2) < 1e-9:
        return np.sin(l1)
    return (np.log(np.cos(l1)) - np.log(np.cos(l2))) / (
        np.log(np.tan(np.pi / 4 + l2 / 2)) - np.log(np.tan(np.pi / 4 + l1 / 2)))


def grid_angle(grid, lon):
    """Angle (rad) between grid north and true north at longitude `lon`."""
    lov = grid["LoVInDegrees"] - 360.0
    d = (np.asarray(lon) - lov + 180.0) % 360.0 - 180.0
    return cone(grid) * np.radians(d)


def grid_to_earth(u, v, grid, lon):
    a = grid_angle(grid, lon)
    c, s = np.cos(a), np.sin(a)
    return c * u + s * v, -s * u + c * v


def earth_to_grid(u, v, grid, lon):
    a = grid_angle(grid, lon)
    c, s = np.cos(a), np.sin(a)
    return c * u - s * v, s * u + c * v


class AnalysisGrid:
    """Every `step`-th HRRR point inside a lon/lat box."""

    def __init__(self, hrrr_grid, extent, step=2):
        self.hgrid = hrrr_grid
        self.step = step
        p = proj_of(hrrr_grid)
        x0, y0 = origin_xy(hrrr_grid)
        dx = hrrr_grid["DxInMetres"]
        w, e, s, n = extent
        lons = np.r_[np.linspace(w, e, 60), np.full(60, e), np.linspace(e, w, 60), np.full(60, w)]
        lats = np.r_[np.full(60, s), np.linspace(s, n, 60), np.full(60, n), np.linspace(n, s, 60)]
        xs, ys = p(lons, lats)
        i0 = max(int(np.floor((min(xs) - x0) / dx)), 0)
        i1 = min(int(np.ceil((max(xs) - x0) / dx)) + 1, hrrr_grid["nx"])
        j0 = max(int(np.floor((min(ys) - y0) / dx)), 0)
        j1 = min(int(np.ceil((max(ys) - y0) / dx)) + 1, hrrr_grid["ny"])
        self.sl = (slice(j0, j1, step), slice(i0, i1, step))
        ii = np.arange(i0, i1, step)
        jj = np.arange(j0, j1, step)
        self.x = x0 + ii * dx
        self.y = y0 + jj * dx
        X, Y = np.meshgrid(self.x, self.y)
        self.lon, self.lat = p(X, Y, inverse=True)
        self.dx = dx * step
        self.shape = X.shape

    def take(self, a):
        """Crop + thin a full HRRR 2-D field (light 3x3 average first, so thinning
        doesn't just sample every other point)."""
        if self.step > 1:
            from scipy.ndimage import uniform_filter
            a = uniform_filter(np.nan_to_num(a, nan=np.nanmean(a)), size=self.step + 1, mode="nearest")
        return a[self.sl].astype(np.float32)

    def remapper(self, src_grid):
        """Function that bilinearly maps a full src-grid field onto this grid."""
        p = proj_of(src_grid)
        x0, y0 = origin_xy(src_grid)
        xs, ys = p(self.lon, self.lat)
        fi = (xs - x0) / src_grid["DxInMetres"]
        fj = (ys - y0) / src_grid["DyInMetres"]
        coords = np.array([fj.ravel(), fi.ravel()])
        shape = self.shape

        def remap(a):
            return map_coordinates(a, coords, order=1, mode="nearest").reshape(shape).astype(np.float32)
        return remap
