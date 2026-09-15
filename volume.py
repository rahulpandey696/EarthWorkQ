# -*- coding: utf-8 -*-
"""
Volume computation for EarthWorkQ.

Two independent methods are provided so that the answer can be checked rather
than merely produced:

  1. Grid integration with fractional boundary weights.  Fast, works with any
     design surface, and is the method used for all reported quantities.

  2. Exact TIN prismatic integration.  The difference between the existing
     surface and a planar design is linear over each Delaunay triangle, so the
     cut and fill volumes can be integrated in closed form, including the
     partial triangles that straddle the zero line.  Available only when the
     design surface is planar, but where it applies it is exact and is the
     ideal cross-check on the grid result.

In addition a grid convergence test recomputes the quantities at half and a
quarter of the chosen cell size.  If the volume is still moving as the grid is
refined, the cell size is too coarse and the report says so.
"""

from __future__ import annotations

import math

import numpy as np


class VolumeResult(object):
    """Cut and fill for one polygon, or for the whole site."""

    def __init__(self, cut=0.0, fill=0.0, area=0.0, cut_area=0.0,
                 fill_area=0.0, max_cut=0.0, max_fill=0.0, mean_depth=0.0,
                 nodata_area=0.0, label=None):
        self.label = label
        self.cut = float(cut)
        self.fill = float(fill)
        self.area = float(area)
        self.cut_area = float(cut_area)
        self.fill_area = float(fill_area)
        self.max_cut = float(max_cut)
        self.max_fill = float(max_fill)
        self.mean_depth = float(mean_depth)
        self.nodata_area = float(nodata_area)

    @property
    def net(self):
        """Positive means surplus material (more cut than fill)."""
        return self.cut - self.fill

    def net_after_shrinkage(self, soil=None):
        if soil is None:
            return self.net
        return self.cut - soil.cut_for_fill(self.fill)

    @property
    def coverage_fraction(self):
        total = self.area + self.nodata_area
        return self.area / total if total > 0 else 0.0

    def as_dict(self):
        return dict(label=self.label, cut=self.cut, fill=self.fill,
                    net=self.net, area=self.area, cut_area=self.cut_area,
                    fill_area=self.fill_area, max_cut=self.max_cut,
                    max_fill=self.max_fill, mean_depth=self.mean_depth,
                    nodata_area=self.nodata_area)


def depth_array(ground, design):
    """Signed depth: positive = fill required, negative = cut required."""
    return np.asarray(design, float) - np.asarray(ground, float)


def integrate(depth, weight, cell_area, label=None):
    """Integrate a signed depth field over fractional cell weights."""
    depth = np.asarray(depth, float)
    weight = np.asarray(weight, float)

    valid = np.isfinite(depth) & (weight > 0)
    nodata = (~np.isfinite(depth)) & (weight > 0)

    w = np.where(valid, weight, 0.0)
    d = np.where(valid, depth, 0.0)

    fill_depth = np.clip(d, 0.0, None)
    cut_depth = np.clip(-d, 0.0, None)

    fill_vol = float((fill_depth * w).sum() * cell_area)
    cut_vol = float((cut_depth * w).sum() * cell_area)
    area = float(w.sum() * cell_area)
    nodata_area = float(weight[nodata].sum() * cell_area) if nodata.any() else 0.0

    cut_area = float(w[cut_depth > 0].sum() * cell_area)
    fill_area = float(w[fill_depth > 0].sum() * cell_area)
    mean_depth = float((d * w).sum() / w.sum()) if w.sum() > 0 else 0.0

    return VolumeResult(cut=cut_vol, fill=fill_vol, area=area,
                        cut_area=cut_area, fill_area=fill_area,
                        max_cut=float(cut_depth[valid].max()) if valid.any() else 0.0,
                        max_fill=float(fill_depth[valid].max()) if valid.any() else 0.0,
                        mean_depth=mean_depth, nodata_area=nodata_area,
                        label=label)


def integrate_per_feature(depth, weights_by_id, mapping, cell_area):
    """One VolumeResult per area-of-interest polygon."""
    out = {}
    for dense_id, w in weights_by_id.items():
        out[dense_id] = integrate(depth, w, cell_area,
                                  label="Feature %s" % mapping.get(dense_id, dense_id))
    return out


# ----------------------------------------------------------------------
# exact TIN integration
# ----------------------------------------------------------------------
def tin_volume(x, y, z, design, clip_geometry=None):
    """Exact cut and fill against a planar design, integrated over triangles.

    Over each Delaunay triangle both the ground (linear TIN) and a planar
    design vary linearly, so f = z_design - z_ground is linear too.  The signed
    volume over a triangle is therefore area * mean(f) at the three vertices.
    Where f changes sign inside a triangle the triangle is split at the zero
    line: the sub-triangle containing the odd vertex has two vertices at f = 0,
    so its volume is simply its area times f_odd / 3.
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        return None

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    pts = np.column_stack((x, y))
    try:
        tri = Delaunay(pts)
    except Exception:
        return None

    if hasattr(design, "evaluate"):
        zd = np.asarray(design.evaluate(x, y), float)
    elif hasattr(design, "level"):
        zd = np.full(x.shape, float(design.level))
    else:
        return None

    f = zd - z                     # positive = fill
    simplices = tri.simplices

    if clip_geometry is not None:
        centroids_x = x[simplices].mean(axis=1)
        centroids_y = y[simplices].mean(axis=1)
        keep = _points_in_geometry(centroids_x, centroids_y, clip_geometry)
        simplices = simplices[keep]
        if simplices.size == 0:
            return None

    p = pts[simplices]             # (n, 3, 2)
    fv = f[simplices]              # (n, 3)

    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))

    signed = area * fv.mean(axis=1)     # fill positive, cut negative

    pos = fv > 0
    n_pos = pos.sum(axis=1)

    fill = np.zeros(area.shape)
    cut = np.zeros(area.shape)

    all_pos = n_pos == 3
    all_neg = n_pos == 0
    fill[all_pos] = signed[all_pos]
    cut[all_neg] = -signed[all_neg]

    mixed = ~(all_pos | all_neg)
    if mixed.any():
        idx = np.where(mixed)[0]
        for i in idx:
            fv_i = fv[i]
            p_i = p[i]
            odd_is_pos = n_pos[i] == 1
            odd = int(np.argmax(fv_i > 0)) if odd_is_pos else int(np.argmin(fv_i > 0))
            others = [j for j in range(3) if j != odd]
            f_odd = fv_i[odd]
            corners = [p_i[odd]]
            for j in others:
                t = f_odd / (f_odd - fv_i[j])
                corners.append(p_i[odd] + t * (p_i[j] - p_i[odd]))
            c = np.array(corners)
            a_small = 0.5 * abs((c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) -
                                (c[2, 0] - c[0, 0]) * (c[1, 1] - c[0, 1]))
            v_odd = a_small * f_odd / 3.0
            if odd_is_pos:
                fill[i] = v_odd
                cut[i] = v_odd - signed[i]
            else:
                cut[i] = -v_odd
                fill[i] = signed[i] + cut[i]

    return VolumeResult(cut=float(cut.sum()), fill=float(fill.sum()),
                        area=float(area.sum()), label="TIN prismatic")


def _points_in_geometry(xs, ys, geometry):
    from qgis.core import QgsGeometry, QgsPointXY
    engine = QgsGeometry.createGeometryEngine(geometry.constGet())
    engine.prepareGeometry()
    keep = np.zeros(xs.shape, dtype=bool)
    for i, (a, b) in enumerate(zip(xs, ys)):
        keep[i] = engine.intersects(
            QgsGeometry.fromPointXY(QgsPointXY(float(a), float(b))).constGet())
    return keep


# ----------------------------------------------------------------------
# convergence
# ----------------------------------------------------------------------
def convergence_test(compute_at_cell, base_cell, factors=(1, 2, 4)):
    """Recompute quantities on progressively finer grids.

    ``compute_at_cell`` takes a cell size and returns a VolumeResult.  Returns
    a list of (cell_size, result) plus a verdict on whether the quantities have
    stabilised.
    """
    rows = []
    for f in factors:
        cell = base_cell / float(f)
        try:
            res = compute_at_cell(cell)
        except Exception:
            continue
        rows.append((cell, res))
    if len(rows) < 2:
        return rows, None

    finest = rows[-1][1]
    coarsest = rows[0][1]
    verdict = {}
    for name in ("cut", "fill"):
        v_fine = getattr(finest, name)
        v_coarse = getattr(coarsest, name)
        if v_fine > 0:
            verdict[name] = abs(v_coarse - v_fine) / v_fine * 100.0
        else:
            verdict[name] = 0.0
    verdict["converged"] = max(verdict["cut"], verdict["fill"]) < 1.0
    return rows, verdict


def volume_uncertainty(rmse, area, n_points):
    """First-order uncertainty on a volume from the interpolation RMSE.

    Errors are treated as spatially uncorrelated at the scale of the point
    spacing, which gives sigma_V ~ RMSE * A / sqrt(n).  This is deliberately
    an order-of-magnitude statement, not a rigorous geostatistical variance.
    """
    if not rmse or n_points <= 0 or area <= 0:
        return None
    return float(rmse * area / math.sqrt(n_points))
