# -*- coding: utf-8 -*-
"""
Output products for EarthWorkQ: rasters, zone polygons, depth contours,
result tables and the balance chart.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

from . import compat
from .grid import NODATA, write_raster


# ----------------------------------------------------------------------
# rasters
# ----------------------------------------------------------------------
def write_depth_raster(path, depth, weight, grid, mask_outside=True):
    """Signed depth raster: positive = fill, negative = cut."""
    arr = np.array(depth, dtype=float, copy=True)
    if mask_outside:
        arr = np.where(np.asarray(weight) > 0, arr, np.nan)
    return write_raster(path, arr, grid)


def write_surface_raster(path, surface, grid):
    return write_raster(path, surface, grid)


DIVERGING_STOPS = [
    (-1.00, "#8c2d04", "Deep cut"),
    (-0.60, "#d94801", ""),
    (-0.30, "#fd8d3c", ""),
    (-0.08, "#fdd0a2", ""),
    (0.00, "#f7f7f7", "Balanced"),
    (0.08, "#c6dbef", ""),
    (0.30, "#6baed6", ""),
    (0.60, "#2171b5", ""),
    (1.00, "#08306b", "Deep fill"),
]


def apply_cut_fill_style(layer, symmetric_max=None):
    """Diverging ramp centred on zero, so cut and fill read at a glance.

    Every symbology class is imported here rather than at module scope. Raster
    shading classes have been renamed and removed between QGIS releases, and a
    styling detail must never be able to stop the plugin from loading - an
    unstyled correct raster is a far better outcome than no plugin at all.
    """
    from qgis.core import (
        QgsColorRampShader,
        QgsRasterShader,
        QgsSingleBandPseudoColorRenderer,
    )
    from qgis.PyQt.QtGui import QColor

    provider = layer.dataProvider()
    stats = provider.bandStatistics(1)
    lo, hi = float(stats.minimumValue), float(stats.maximumValue)
    m = symmetric_max or max(abs(lo), abs(hi), 1e-6)
    m = float(np.ceil(m * 100.0) / 100.0)

    shader_fn = QgsColorRampShader()
    interpolated = compat.colour_ramp_interpolated()
    if interpolated is not None:
        shader_fn.setColorRampType(interpolated)
    items = []
    for frac, colour, label in DIVERGING_STOPS:
        value = frac * m
        text = label if label else ("%+.2f" % value)
        items.append(QgsColorRampShader.ColorRampItem(value, QColor(colour), text))
    shader_fn.setColorRampItemList(items)

    shader = QgsRasterShader()
    shader.setRasterShaderFunction(shader_fn)
    renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
    renderer.setClassificationMin(-m)
    renderer.setClassificationMax(m)
    layer.setRenderer(renderer)
    layer.triggerRepaint()
    return layer


def load_raster(path, name, style_cut_fill=False, add_to_project=True):
    layer = QgsRasterLayer(path, name)
    if not layer.isValid():
        return None
    if style_cut_fill:
        try:
            apply_cut_fill_style(layer)
        except Exception:
            # styling is cosmetic; never let it fail the run
            pass
    if add_to_project:
        QgsProject.instance().addMapLayer(layer)
    return layer


# ----------------------------------------------------------------------
# zone polygons and contours
# ----------------------------------------------------------------------
def write_zone_polygons(path, depth, weight, grid, tolerance=0.05,
                        add_to_project=True):
    """Polygonise cut / balanced / fill zones with an area attribute."""
    import processing

    cls = np.full(grid.shape, np.nan)
    d = np.asarray(depth, float)
    w = np.asarray(weight, float)
    inside = (w > 0) & np.isfinite(d)
    cls[inside & (d < -tolerance)] = -1
    cls[inside & (np.abs(d) <= tolerance)] = 0
    cls[inside & (d > tolerance)] = 1

    tmp = os.path.splitext(path)[0] + "_zones_tmp.tif"
    write_raster(tmp, cls, grid, nodata=NODATA)
    res = processing.run("gdal:polygonize", {
        "INPUT": tmp, "BAND": 1, "FIELD": "zone_code",
        "EIGHT_CONNECTEDNESS": False, "EXTRA": "", "OUTPUT": path,
    })
    try:
        os.remove(tmp)
    except OSError:
        pass

    layer = QgsVectorLayer(res["OUTPUT"], "Cut and fill zones", "ogr")
    if layer.isValid() and add_to_project:
        QgsProject.instance().addMapLayer(layer)
    return layer


def write_depth_contours(path, depth_raster_path, interval,
                         add_to_project=True):
    """Contours of the difference surface - useful for setting out."""
    import processing
    res = processing.run("gdal:contour", {
        "INPUT": depth_raster_path, "BAND": 1, "INTERVAL": float(interval),
        "FIELD_NAME": "depth", "CREATE_3D": False, "IGNORE_NODATA": False,
        "NODATA": NODATA, "OFFSET": 0, "EXTRA": "", "OUTPUT": path,
    })
    layer = QgsVectorLayer(res["OUTPUT"], "Cut and fill depth contours", "ogr")
    if layer.isValid() and add_to_project:
        QgsProject.instance().addMapLayer(layer)
    return layer


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------
def write_results_csv(path, per_feature, mapping, soil=None, unit="m"):
    header = ["feature_id", "cut_%s3" % unit, "fill_%s3" % unit,
              "net_%s3" % unit, "area_%s2" % unit,
              "cut_area_%s2" % unit, "fill_area_%s2" % unit,
              "max_cut_%s" % unit, "max_fill_%s" % unit,
              "mean_depth_%s" % unit, "uncovered_area_%s2" % unit]
    if soil is not None:
        header += ["cut_required_for_fill_%s3" % unit,
                   "surplus_%s3" % unit]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for dense_id in sorted(per_feature):
            r = per_feature[dense_id]
            row = [mapping.get(dense_id, dense_id),
                   round(r.cut, 3), round(r.fill, 3), round(r.net, 3),
                   round(r.area, 3), round(r.cut_area, 3),
                   round(r.fill_area, 3), round(r.max_cut, 3),
                   round(r.max_fill, 3), round(r.mean_depth, 4),
                   round(r.nodata_area, 3)]
            if soil is not None:
                need = soil.cut_for_fill(r.fill)
                row += [round(need, 3), round(r.cut - need, 3)]
            writer.writerow(row)
    return path


def write_curve_csv(path, curve):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for row in curve.to_rows():
            writer.writerow([round(v, 4) if isinstance(v, float) else v
                             for v in row])
    return path


def write_results_to_layer(aoi_layer, per_feature, mapping, soil=None,
                           prefix="ewq", decimals=2):
    """Write quantities back onto the area-of-interest layer as attributes."""
    from qgis.core import edit

    dp = aoi_layer.dataProvider()
    if not compat.provider_can(dp, "AddAttributes"):
        return False, ("The area of interest layer does not accept new "
                       "attributes, so quantities could not be written back. "
                       "The CSV output still contains them.")

    fields = [("%s_cut" % prefix, "Cut volume"),
              ("%s_fill" % prefix, "Fill volume"),
              ("%s_net" % prefix, "Net (cut - fill)"),
              ("%s_area" % prefix, "Computed area")]
    if soil is not None:
        fields.append(("%s_surp" % prefix, "Surplus after shrinkage"))

    existing = [f.name() for f in aoi_layer.fields()]
    to_add = [compat.make_field(n, "double") for n, _ in fields if n not in existing]
    if to_add:
        dp.addAttributes(to_add)
        aoi_layer.updateFields()

    idx = {n: aoi_layer.fields().lookupField(n) for n, _ in fields}
    reverse = {v: k for k, v in mapping.items()}

    with edit(aoi_layer):
        for feat in aoi_layer.getFeatures():
            dense = reverse.get(feat.id())
            if dense is None or dense not in per_feature:
                continue
            r = per_feature[dense]
            feat.setAttribute(idx["%s_cut" % prefix], round(r.cut, decimals))
            feat.setAttribute(idx["%s_fill" % prefix], round(r.fill, decimals))
            feat.setAttribute(idx["%s_net" % prefix], round(r.net, decimals))
            feat.setAttribute(idx["%s_area" % prefix], round(r.area, decimals))
            if soil is not None:
                surp = r.cut - soil.cut_for_fill(r.fill)
                feat.setAttribute(idx["%s_surp" % prefix], round(surp, decimals))
            aoi_layer.updateFeature(feat)
    return True, None


# ----------------------------------------------------------------------
# chart
# ----------------------------------------------------------------------
def write_balance_chart(path, curve, balance_level=None, chosen_level=None,
                        title="Cut and fill against formation level",
                        unit="m"):
    """Render the balance curve. Returns the path, or None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    ax1.plot(curve.offsets, curve.cut, color="#c0392b", lw=2, label="Cut")
    ax1.plot(curve.offsets, curve.fill, color="#1f5fa9", lw=2, label="Fill")
    ax1.plot(curve.offsets, curve.total_movement, color="#7f8c8d", lw=1.2,
             ls="--", label="Total movement")
    ax1.set_ylabel("Volume (%s3)" % unit)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper center", frameon=False, ncol=3)
    ax1.set_title(title)

    ax2.axhline(0.0, color="#333333", lw=1)
    ax2.plot(curve.offsets, curve.net, color="#27682a", lw=2,
             label="Net after shrinkage (cut - SF x fill)")
    ax2.fill_between(curve.offsets, curve.net, 0,
                     where=(curve.net >= 0), color="#c0392b", alpha=0.18,
                     label="Surplus")
    ax2.fill_between(curve.offsets, curve.net, 0,
                     where=(curve.net < 0), color="#1f5fa9", alpha=0.18,
                     label="Deficit")
    ax2.set_ylabel("Net (%s3)" % unit)
    ax2.set_xlabel(curve.reference_label + " (%s)" % unit)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", frameon=False, fontsize=8)

    for ax in (ax1, ax2):
        if balance_level is not None:
            ax.axvline(balance_level, color="#27682a", ls=":", lw=1.6)
        if chosen_level is not None:
            ax.axvline(chosen_level, color="#8e44ad", ls="-.", lw=1.4)

    if balance_level is not None:
        ax1.annotate("balance at %.3f" % balance_level,
                     xy=(balance_level, ax1.get_ylim()[1] * 0.92),
                     xytext=(5, 0), textcoords="offset points",
                     color="#27682a", fontsize=9)
    if chosen_level is not None:
        ax1.annotate("selected %.3f" % chosen_level,
                     xy=(chosen_level, ax1.get_ylim()[1] * 0.80),
                     xytext=(5, 0), textcoords="offset points",
                     color="#8e44ad", fontsize=9)

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def write_hypsometry_chart(path, ground, weight, cell_area, unit="m"):
    """Area-elevation curve of the site - context for the balance level."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    g = np.asarray(ground, float)
    w = np.asarray(weight, float)
    valid = np.isfinite(g) & (w > 0)
    if not valid.any():
        return None
    gv, wv = g[valid], w[valid]
    order = np.argsort(gv)
    gv, wv = gv[order], wv[order]
    cum_area = np.cumsum(wv) * cell_area

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cum_area, gv, color="#8a5a2b", lw=2)
    ax.set_xlabel("Cumulative area below elevation (%s2)" % unit)
    ax.set_ylabel("Ground elevation (%s)" % unit)
    ax.set_title("Site area-elevation (hypsometric) curve")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
