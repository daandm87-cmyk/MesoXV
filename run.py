"""MesoXV driver.

Replay a past day:   python run.py --start 2026-04-17T19:00 --end 2026-04-18T01:00 --regions iowa,midwest
Live (chase day):    python run.py --live --hours 6          (keeps going every 15 minutes)

Output (default docs/): index.html viewer, frames.json, img/<YYYYmmddHHMM>/<region>/<product>.png
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = dt.timedelta(minutes=15)
KEEP_HOURS = 3                       # frames shown in the viewer (live mode)


# ------------------------------------------------------------------ time labels
def _nth_sunday(year, month, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(6 - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def central(t_utc):
    """(local datetime, 'CDT'/'CST') for US Central time, from a naive UTC datetime."""
    y = t_utc.year
    start = dt.datetime.combine(_nth_sunday(y, 3, 2), dt.time(8))    # 2 AM CST = 08 UTC
    end = dt.datetime.combine(_nth_sunday(y, 11, 1), dt.time(7))     # 2 AM CDT = 07 UTC
    if start <= t_utc < end:
        return t_utc - dt.timedelta(hours=5), "CDT"
    return t_utc - dt.timedelta(hours=6), "CST"


def label(t):
    lt, tz = central(t)
    return "%s %s  (%sZ)" % (lt.strftime("%a %b %d  %I:%M %p").replace(" 0", " "), tz, t.strftime("%H%M"))


# ------------------------------------------------------------------ rendering worker
def _render_frame(args):
    frame_d, grid, valid, regions, products, out = args
    import render
    done = 0
    src = "RTMA Rapid Update %sZ analysis + HRRR %sZ run" % (valid.strftime("%H%M"), frame_d["_run"].strftime("%H"))
    if frame_d.get("_refl_time") is not None:
        src += "  ·  radar outlines 35/50 dBZ: MRMS %sZ" % frame_d["_refl_time"].strftime("%H%M")
    src += "  ·  MesoXV"
    base = os.path.join(out, "img", valid.strftime("%Y%m%d%H%M"))
    skipped = {}
    for reg in regions:
        folder = os.path.join(base, reg)
        os.makedirs(folder, exist_ok=True)
        for prod in products:
            ok, why = render.worth_showing(frame_d, grid, prod, reg)
            path = os.path.join(folder, prod + ".png")
            if not ok:                      # nothing worth drawing: say why on the page instead
                skipped.setdefault(reg, {})[prod] = why
                if os.path.exists(path):
                    os.remove(path)
                continue
            png = render.render(frame_d, grid, prod, reg, label(valid), src)
            with open(path, "wb") as fh:
                fh.write(png)
            done += 1
    with open(os.path.join(base, "skip.json"), "w", encoding="utf-8") as fh:
        json.dump(skipped, fh)
    return valid, done


# ------------------------------------------------------------------ main
def compute_extent(regions):
    import render
    ex = [render.REGIONS[r][1] for r in regions]
    return (min(e[0] for e in ex) - 1.5, max(e[1] for e in ex) + 1.5,
            min(e[2] for e in ex) - 1.0, max(e[3] for e in ex) + 1.0)


def add_tendencies(d, past):
    """1-hour changes, from the frame one hour earlier if we have it."""
    old = past.get(d["_valid"] - dt.timedelta(hours=1))
    if old is None:
        return
    d["dmslp1h"] = d["mslp"] - old["mslp"]
    d["dmlcape1h"] = d["mlcape"] - old["mlcape"]
    # only where the air was unstable both then and now: a passing front otherwise
    # paints a stripe of "cap strengthening" that hides the warm-sector signal
    both = (d["mlcape"] > 250) & (old["mlcape"] > 250) & np.isfinite(d["mlcin"]) & np.isfinite(old["mlcin"])
    d["dmlcin1h"] = np.where(both, d["mlcin"] - old["mlcin"], np.nan)


def write_index(out, regions, products, live):
    import render
    frames = sorted(n for n in os.listdir(os.path.join(out, "img")) if n.isdigit())

    def skips(f):
        try:
            with open(os.path.join(out, "img", f, "skip.json"), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    meta = {
        "frames": [{"id": f, "label": label(dt.datetime.strptime(f, "%Y%m%d%H%M")), "skip": skips(f)}
                   for f in frames],
        "regions": [{"id": r, "name": render.REGIONS[r][0]} for r in regions],
        "products": [{"id": p, "name": render.PRODUCTS[p]["title"]} for p in products],
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "live": live,
    }
    with open(os.path.join(out, "frames.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    shutil.copy2(os.path.join(HERE, "viewer.html"), os.path.join(out, "index.html"))
    open(os.path.join(out, ".nojekyll"), "w").close()        # GitHub Pages: serve as-is, no Jekyll


def main():
    import render
    import frame as fr
    ap = argparse.ArgumentParser(description="MesoXV: 15-minute mesoanalysis (RTMA-RU surface + HRRR aloft)")
    ap.add_argument("--start", help="first frame, UTC (YYYY-MM-DDTHH:MM)")
    ap.add_argument("--end", help="last frame, UTC")
    ap.add_argument("--live", action="store_true", help="follow real time, a new frame every 15 minutes")
    ap.add_argument("--hours", type=float, default=6, help="live: how long to keep running")
    ap.add_argument("--regions", default=",".join(render.REGIONS))
    ap.add_argument("--products", default=",".join(render.ORDER))
    ap.add_argument("--out", default=os.path.join(HERE, "docs"))
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()
    regions = a.regions.split(",")
    products = a.products.split(",")
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    inp = fr.Inputs(compute_extent(regions), step=2)
    past = {}

    def one(valid, render_it=True):
        t0 = time.time()
        d = fr.make_frame(inp, valid)
        d["_valid"] = valid
        add_tendencies(d, past)
        past[valid] = {k: d[k] for k in ("mslp", "mlcape", "mlcin")}
        for old in [t for t in past if t < valid - dt.timedelta(hours=1, minutes=5)]:
            past.pop(old)
        print("%s  analysed in %.0f s%s" % (valid.strftime("%Y-%m-%d %H:%MZ"), time.time() - t0,
                                            "" if render_it else "  (warm-up for 1-h changes)"), flush=True)
        return d

    if a.live:
        run_live(a, inp, one, regions, products)
        return

    start = dt.datetime.fromisoformat(a.start)
    end = dt.datetime.fromisoformat(a.end)
    jobs = []
    t = start - dt.timedelta(hours=1)
    while t <= end:
        try:
            d = one(t, t >= start)
        except Exception as e:
            print("%s  skipped: %s" % (t.strftime("%H:%MZ"), e), flush=True)
            t += STEP
            continue
        if t >= start:
            d.pop("_check", None)
            jobs.append((d, inp.grid, t, regions, products, a.out))
        t += STEP
    print("rendering %d frames x %d regions x %d products with %d workers ..." % (
        len(jobs), len(regions), len(products), a.workers), flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for valid, n in ex.map(_render_frame, jobs):
            print("  %s: %d maps" % (valid.strftime("%H:%MZ"), n), flush=True)
    print("rendered in %.0f s" % (time.time() - t0))
    write_index(a.out, regions, products, live=False)
    print("viewer: %s" % os.path.join(a.out, "index.html"))
    publish()


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def publish():
    """Run the publish hook (the GitHub workflow sets it to push the site)."""
    cmd = os.environ.get("MESOXV_PUBLISH")
    if cmd:
        rc = os.system(cmd)
        print("  published" if rc == 0 else "  publish failed (%s)" % rc, flush=True)


def have_frame(out, t, regions):
    base = os.path.join(out, "img", t.strftime("%Y%m%d%H%M"))
    return all(os.path.isdir(os.path.join(base, r)) for r in regions)


def run_live(a, inp, one, regions, products):
    import frame as fr
    stop_at = utcnow() + dt.timedelta(hours=a.hours)
    newest = (utcnow() - fr.RTMA_DELAY).replace(second=0, microsecond=0)
    newest -= dt.timedelta(minutes=newest.minute % 15)
    # warm up: analyse the last 1 h 45 min and draw the last hour's frames the site
    # doesn't have yet (each drawn frame then has its 1-hour-earlier frame for the
    # change maps), so the loop isn't empty right after a start
    t = newest - dt.timedelta(hours=1, minutes=45)
    nxt = newest + STEP
    while t <= newest:
        try:
            d = one(t)
            if t > newest - dt.timedelta(hours=1) and not have_frame(a.out, t, regions):
                d.pop("_check", None)
                _render_frame((d, inp.grid, t, regions, products, a.out))
        except Exception as e:
            print("%s  skipped: %s" % (t.strftime("%H:%MZ"), e), flush=True)
            if t == newest:
                nxt = newest                       # probably not posted yet: retry in the loop
        t += STEP
    prune(a.out)
    write_index(a.out, regions, products, live=True)
    publish()
    while utcnow() < stop_at:
        if utcnow() < nxt + fr.RTMA_DELAY - dt.timedelta(minutes=4):
            time.sleep(30)                           # not due yet
            continue
        try:
            d = one(nxt)
        except Exception as e:                       # usually: RTMA not posted yet
            if utcnow() - nxt > dt.timedelta(minutes=40):
                print("%s  gave up: %s" % (nxt.strftime("%H:%MZ"), e), flush=True)
                nxt += STEP
            else:
                time.sleep(60)
            continue
        d.pop("_check", None)
        _render_frame((d, inp.grid, nxt, regions, products, a.out))
        prune(a.out)
        write_index(a.out, regions, products, live=True)
        publish()
        nxt += STEP


def prune(out):
    """Live: keep the last KEEP_HOURS of frames (by time, so a replay left on the site goes too)."""
    frames = sorted(n for n in os.listdir(os.path.join(out, "img")) if n.isdigit())
    if not frames:
        return
    newest = dt.datetime.strptime(frames[-1], "%Y%m%d%H%M")
    for f in frames:
        if dt.datetime.strptime(f, "%Y%m%d%H%M") < newest - dt.timedelta(hours=KEEP_HOURS):
            shutil.rmtree(os.path.join(out, "img", f), ignore_errors=True)


if __name__ == "__main__":
    main()
