# -*- coding: utf-8 -*-
"""
Elevation input adapters for EarthWorkQ.

Four kinds of survey input are supported.  Every one of them is reduced to a
common intermediate form - either a scattered point cloud (x, y, z) or a
raster grid - so that nothing downstream needs to know where the data came
from.

    CSV / XLSX  ->  ScatterSource
    Point layer ->  ScatterSource
    Contours    ->  ContourSource  (a ScatterSource with extra topology checks)
    DEM raster  ->  RasterSource
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsWkbTypes,
)

from . import compat


class SourceError(ValueError):
    pass


class QualityIssue(object):
    """A single input-quality finding, carried through to the report."""

    SEVERITY_INFO = "info"
    SEVERITY_WARNING = "warning"
    SEVERITY_ERROR = "error"

    def __init__(self, severity, message):
        self.severity = severity
        self.message = message

    def __str__(self):
        return "[%s] %s" % (self.severity.upper(), self.message)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def check_projected_crs(crs, what):
    """Earthwork volumes are meaningless in degrees. Enforce a metric CRS."""
    if crs is None or not crs.isValid():
        raise SourceError(
            "%s has no coordinate reference system defined. Assign a projected "
            "CRS before running EarthWorkQ." % what)
    if crs.isGeographic():
        raise SourceError(
            "%s uses a geographic CRS (%s). Horizontal units are degrees, so "
            "areas and volumes cannot be computed. Reproject to a projected "
            "CRS in metres or feet first." % (what, crs.authid()))
    unit = _unit_name(crs)
    if unit and not any(t in unit for t in ("meter", "metre", "foot", "feet", "ft",
                                            "unknown")):
        raise SourceError(
            "%s uses map units of '%s', which EarthWorkQ cannot convert to a "
            "volume. Use a CRS in metres or feet." % (what, unit))
    return crs


def _unit_name(crs):
    """Lower-case name of the CRS map units, across QGIS versions."""
    try:
        from qgis.core import QgsUnitTypes
        return QgsUnitTypes.encodeUnit(crs.mapUnits()).lower()
    except Exception:
        pass
    try:
        return str(crs.mapUnits().name).lower()
    except Exception:
        return str(crs.mapUnits()).lower()


def horizontal_unit_label(crs):
    unit = _unit_name(crs)
    if any(t in unit for t in ("foot", "feet", "ft")):
        return "ft"
    return "m"


# ----------------------------------------------------------------------
# base class
# ----------------------------------------------------------------------
class ElevationSource(object):
    """Common interface for every kind of elevation input."""

    kind = "abstract"

    def __init__(self, crs, name="elevation"):
        self.crs = crs
        self.name = name
        self.issues = []

    def add_issue(self, severity, message):
        self.issues.append(QualityIssue(severity, message))

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == QualityIssue.SEVERITY_ERROR]

    def extent(self):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError


# ----------------------------------------------------------------------
# scattered points
# ----------------------------------------------------------------------
class ScatterSource(ElevationSource):
    """An (x, y, z) point cloud from a table, a point layer, or contours."""

    kind = "points"

    def __init__(self, x, y, z, crs, name="points"):
        super(ScatterSource, self).__init__(crs, name)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        if not (self.x.size == self.y.size == self.z.size):
            raise SourceError("Coordinate arrays have inconsistent lengths.")
        self._clean()

    # -- cleaning ------------------------------------------------------
    def _clean(self):
        n0 = self.x.size
        finite = np.isfinite(self.x) & np.isfinite(self.y) & np.isfinite(self.z)
        dropped = int(n0 - finite.sum())
        if dropped:
            self.add_issue(
                QualityIssue.SEVERITY_WARNING,
                "%d of %d records had a missing or non-numeric coordinate and "
                "were discarded." % (dropped, n0))
        self.x, self.y, self.z = self.x[finite], self.y[finite], self.z[finite]
        if self.x.size < 3:
            raise SourceError(
                "Only %d usable elevation points were found. At least three "
                "are needed to build a surface." % self.x.size)
        self._check_duplicates()

    def _check_duplicates(self):
        """Coincident XY with differing Z is a survey blunder, not noise."""
        key = np.round(np.column_stack((self.x, self.y)), 4)
        _, idx, inv, counts = np.unique(key, axis=0, return_index=True,
                                        return_inverse=True, return_counts=True)
        n_dup_groups = int((counts > 1).sum())
        if not n_dup_groups:
            return
        conflicting = 0
        for g in np.where(counts > 1)[0]:
            zs = self.z[inv == g]
            if float(zs.max() - zs.min()) > 0.001:
                conflicting += 1
        if conflicting:
            self.add_issue(
                QualityIssue.SEVERITY_ERROR,
                "%d locations carry more than one different elevation "
                "(coincident points with conflicting Z). Resolve these before "
                "computing quantities - they are usually transcription errors."
                % conflicting)
        else:
            self.add_issue(
                QualityIssue.SEVERITY_INFO,
                "%d exactly duplicated points were found and collapsed."
                % (int(counts.sum() - counts.size)))
        # collapse exact duplicates (keeps the interpolators well conditioned)
        self.x, self.y, self.z = self.x[idx], self.y[idx], self.z[idx]

    def flag_outliers(self, k=8, threshold=4.0):
        """Robust local outlier screen using a modified Z-score of the residual
        against the mean of the k nearest neighbours."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return 0
        if self.x.size < k + 1:
            return 0
        pts = np.column_stack((self.x, self.y))
        tree = cKDTree(pts)
        _, nbr = tree.query(pts, k=min(k + 1, self.x.size))
        neigh_mean = self.z[nbr[:, 1:]].mean(axis=1)
        resid = self.z - neigh_mean
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad <= 0:
            return 0
        score = 0.6745 * (resid - med) / mad
        flagged = int((np.abs(score) > threshold).sum())
        if flagged:
            self.add_issue(
                QualityIssue.SEVERITY_WARNING,
                "%d point(s) differ sharply from their neighbours and may be "
                "elevation blunders. They have been retained - review them "
                "before relying on the quantities." % flagged)
        return flagged

    # -- geometry ------------------------------------------------------
    def extent(self):
        return (float(self.x.min()), float(self.y.min()),
                float(self.x.max()), float(self.y.max()))

    def mean_point_spacing(self):
        """Mean nearest-neighbour distance - the natural resolution limit."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            x0, y0, x1, y1 = self.extent()
            area = max((x1 - x0) * (y1 - y0), 1e-9)
            return math.sqrt(area / max(self.x.size, 1))
        pts = np.column_stack((self.x, self.y))
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        return float(np.mean(d[:, 1]))

    def convex_hull(self):
        geom = QgsGeometry.fromMultiPointXY(
            [QgsPointXY(float(a), float(b)) for a, b in zip(self.x, self.y)])
        return geom.convexHull()

    def describe(self):
        x0, y0, x1, y1 = self.extent()
        return [
            ("Source type", "Scattered elevation points"),
            ("Source", self.name),
            ("Points used", "%d" % self.x.size),
            ("Elevation range", "%.3f to %.3f" % (self.z.min(), self.z.max())),
            ("Mean point spacing", "%.2f" % self.mean_point_spacing()),
            ("Extent", "%.1f, %.1f to %.1f, %.1f" % (x0, y0, x1, y1)),
            ("CRS", self.crs.authid()),
        ]


class ContourSource(ScatterSource):
    """Contour polylines, densified into a point cloud.

    Contours are the most error-prone elevation input.  Two failure modes are
    checked for explicitly: crossing contours (corrupt data) and an unrealistic
    accuracy expectation given the contour interval.
    """

    kind = "contours"

    def __init__(self, x, y, z, crs, name="contours", interval=None,
                 geometries=None):
        self.interval = interval
        self._geometries = geometries or []
        super(ContourSource, self).__init__(x, y, z, crs, name)

    def vertical_accuracy(self):
        """Rule of thumb: vertical accuracy is about half the contour interval."""
        return None if not self.interval else self.interval / 2.0

    def describe(self):
        rows = super(ContourSource, self).describe()
        rows[0] = ("Source type", "Contour polylines (densified)")
        if self.interval:
            rows.append(("Detected contour interval", "%.3f" % self.interval))
            rows.append(("Implied vertical accuracy",
                         "+/- %.3f (half the interval)" % self.vertical_accuracy()))
        return rows

    def contour_envelope(self):
        """Union of contour geometries, buffered slightly - the domain outside
        which elevations are genuinely unknown."""
        if not self._geometries:
            return self.convex_hull()
        return self.convex_hull()


# ----------------------------------------------------------------------
# raster
# ----------------------------------------------------------------------
class RasterSource(ElevationSource):
    """An existing DEM."""

    kind = "raster"

    def __init__(self, layer, band=1):
        crs = check_projected_crs(layer.crs(), "The DEM layer '%s'" % layer.name())
        super(RasterSource, self).__init__(crs, layer.name())
        self.layer = layer
        self.band = int(band)
        self._array = None

    def native_cell_size(self):
        return (abs(self.layer.rasterUnitsPerPixelX()),
                abs(self.layer.rasterUnitsPerPixelY()))

    def extent(self):
        e = self.layer.extent()
        return (e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum())

    def source_path(self):
        src = self.layer.source() or ""
        return src.split("|")[0]

    def read_array(self):
        """Read the whole band into a numpy array with nodata as NaN."""
        if self._array is not None:
            return self._array
        from osgeo import gdal
        path = self.source_path()
        ds = gdal.Open(path)
        if ds is None:
            raise SourceError(
                "The DEM '%s' could not be opened directly from disk. In-memory "
                "or database rasters are not supported; export it to GeoTIFF "
                "first." % self.name)
        band = ds.GetRasterBand(self.band)
        arr = band.ReadAsArray().astype(float)
        nd = band.GetNoDataValue()
        if nd is not None:
            arr[np.isclose(arr, nd)] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        ds = None
        self._array = arr
        return arr

    def describe(self):
        cx, cy = self.native_cell_size()
        return [
            ("Source type", "Digital elevation model"),
            ("Source", self.name),
            ("Native cell size", "%.3f x %.3f" % (cx, cy)),
            ("Raster size", "%d x %d" % (self.layer.width(), self.layer.height())),
            ("Band", "%d" % self.band),
            ("CRS", self.crs.authid()),
        ]


# ----------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------
def _sniff_delimiter(sample):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        return ","


def load_table(path, easting_field=None, northing_field=None,
               elevation_field=None, crs=None, sheet=None):
    """Read a CSV / TSV / XLSX table of easting, northing, elevation.

    Field names are matched case-insensitively against a list of common
    aliases when they are not supplied explicitly.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        header, rows = _read_spreadsheet(path, sheet)
    else:
        header, rows = _read_delimited(path)

    lower = [str(h).strip().lower() for h in header]

    def resolve(explicit, aliases, label):
        if explicit:
            if explicit in header:
                return header.index(explicit)
            if explicit.lower() in lower:
                return lower.index(explicit.lower())
            raise SourceError("Column '%s' was not found in %s. Available "
                              "columns: %s" % (explicit, os.path.basename(path),
                                               ", ".join(map(str, header))))
        for a in aliases:
            if a in lower:
                return lower.index(a)
        raise SourceError(
            "Could not identify the %s column in %s. Expected one of: %s. "
            "Select the column explicitly." %
            (label, os.path.basename(path), ", ".join(aliases)))

    ix = resolve(easting_field, ["easting", "east", "x", "e", "eastings", "x_coord", "xcoord"], "easting")
    iy = resolve(northing_field, ["northing", "north", "y", "n", "northings", "y_coord", "ycoord"], "northing")
    iz = resolve(elevation_field, ["elevation", "elev", "z", "level", "rl", "height", "ht", "reduced level"], "elevation")

    xs, ys, zs = [], [], []
    for row in rows:
        if len(row) <= max(ix, iy, iz):
            continue
        try:
            xs.append(float(str(row[ix]).replace(",", "").strip()))
            ys.append(float(str(row[iy]).replace(",", "").strip()))
            zs.append(float(str(row[iz]).replace(",", "").strip()))
        except (TypeError, ValueError):
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)

    if crs is None:
        raise SourceError(
            "A coordinate reference system must be specified for a table of "
            "coordinates - the file itself carries no CRS information.")
    check_projected_crs(crs, "The elevation table")
    src = ScatterSource(xs, ys, zs, crs, name=os.path.basename(path))
    src.add_issue(QualityIssue.SEVERITY_INFO,
                  "Columns used - easting: '%s', northing: '%s', elevation: "
                  "'%s'." % (header[ix], header[iy], header[iz]))
    return src


def table_columns(path, sheet=None):
    """Return the header row so the GUI can offer a field chooser."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        header, _ = _read_spreadsheet(path, sheet, max_rows=1)
    else:
        header, _ = _read_delimited(path, max_rows=1)
    return [str(h) for h in header]


def _read_delimited(path, max_rows=None):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        reader = csv.reader(fh, delimiter=_sniff_delimiter(sample))
        rows = []
        header = None
        for i, row in enumerate(reader):
            if not row:
                continue
            if header is None:
                header = row
                continue
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    if header is None:
        raise SourceError("The file %s appears to be empty." % os.path.basename(path))
    return header, rows


def _read_spreadsheet(path, sheet=None, max_rows=None):
    try:
        import openpyxl
    except ImportError:
        raise SourceError(
            "Reading .xlsx files requires the 'openpyxl' package, which is not "
            "available in this QGIS installation. Save the sheet as CSV and "
            "load that instead, or install openpyxl via the OSGeo4W shell "
            "(python -m pip install openpyxl).")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb[wb.sheetnames[0]]
    header = None
    rows = []
    for r in ws.iter_rows(values_only=True):
        if r is None or all(v is None for v in r):
            continue
        if header is None:
            header = list(r)
            continue
        rows.append(list(r))
        if max_rows and len(rows) >= max_rows:
            break
    wb.close()
    if header is None:
        raise SourceError("The worksheet in %s is empty." % os.path.basename(path))
    return header, rows


def load_point_layer(layer, elevation_field, use_z=False):
    """Read a point vector layer, taking Z either from an attribute or geometry."""
    crs = check_projected_crs(layer.crs(), "The point layer '%s'" % layer.name())
    if not compat.is_point_layer(layer):
        raise SourceError("'%s' is not a point layer." % layer.name())

    has_z = compat.has_z(layer)
    if use_z and not has_z:
        raise SourceError(
            "'%s' has no Z values in its geometry. Choose an elevation "
            "attribute instead." % layer.name())
    if not use_z:
        if elevation_field not in [f.name() for f in layer.fields()]:
            raise SourceError("Field '%s' does not exist in '%s'."
                              % (elevation_field, layer.name()))

    xs, ys, zs = [], [], []
    for feat in layer.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        for part in geom.constGet().vertices() if geom.isMultipart() else [geom.constGet()]:
            try:
                px, py = float(part.x()), float(part.y())
            except AttributeError:
                continue
            if use_z:
                pz = float(part.z())
            else:
                val = feat[elevation_field]
                try:
                    pz = float(val)
                except (TypeError, ValueError):
                    pz = np.nan
            xs.append(px)
            ys.append(py)
            zs.append(pz)

    src = ScatterSource(xs, ys, zs, crs, name=layer.name())
    src.add_issue(QualityIssue.SEVERITY_INFO,
                  "Elevation taken from %s." %
                  ("geometry Z values" if use_z else "attribute '%s'" % elevation_field))
    return src


def load_contour_layer(layer, elevation_field, densify_spacing=None,
                       check_topology=True):
    """Densify contour polylines into points, with topology validation."""
    crs = check_projected_crs(layer.crs(), "The contour layer '%s'" % layer.name())
    if not compat.is_line_layer(layer):
        raise SourceError("'%s' is not a line layer." % layer.name())
    if elevation_field not in [f.name() for f in layer.fields()]:
        raise SourceError("Field '%s' does not exist in '%s'."
                          % (elevation_field, layer.name()))

    raw = []
    for feat in layer.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            elev = float(feat[elevation_field])
        except (TypeError, ValueError):
            continue
        raw.append((QgsGeometry(geom), elev))

    if not raw:
        raise SourceError(
            "No contour features with a usable elevation were found in '%s'."
            % layer.name())

    # spacing default: a fifth of the median vertex spacing along the lines
    if densify_spacing is None:
        lengths = [g.length() for g, _ in raw if g.length() > 0]
        counts = [len(list(g.vertices())) for g, _ in raw]
        total_len = float(np.sum(lengths)) if lengths else 0.0
        total_v = float(np.sum(counts)) if counts else 1.0
        densify_spacing = max(total_len / max(total_v, 1.0) * 0.5, 0.1)

    xs, ys, zs = [], [], []
    geoms = []
    for geom, elev in raw:
        dens = geom.densifyByDistance(densify_spacing)
        geoms.append((dens, elev))
        for v in dens.vertices():
            xs.append(v.x())
            ys.append(v.y())
            zs.append(elev)

    interval = _detect_interval(sorted(set(e for _, e in raw)))
    src = ContourSource(xs, ys, zs, crs, name=layer.name(),
                        interval=interval, geometries=geoms)
    src.add_issue(
        QualityIssue.SEVERITY_INFO,
        "%d contour lines densified at %.2f spacing into %d vertices."
        % (len(raw), densify_spacing, len(xs)))

    if check_topology:
        _check_contour_topology(raw, src)
    return src


def _detect_interval(levels):
    if len(levels) < 2:
        return None
    diffs = np.diff(np.array(levels, dtype=float))
    diffs = diffs[diffs > 1e-6]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _check_contour_topology(raw, src, max_pairs=200000):
    """Contours of different elevations must never cross."""
    crossings = 0
    self_int = 0
    n = len(raw)
    boxes = [g.boundingBox() for g, _ in raw]
    pairs_checked = 0
    for i in range(n):
        gi, ei = raw[i]
        if not gi.isGeosValid():
            self_int += 1
        for j in range(i + 1, n):
            if pairs_checked > max_pairs:
                break
            if not boxes[i].intersects(boxes[j]):
                continue
            pairs_checked += 1
            gj, ej = raw[j]
            if abs(ei - ej) < 1e-9:
                continue
            if gi.crosses(gj):
                crossings += 1
    if crossings:
        src.add_issue(
            QualityIssue.SEVERITY_ERROR,
            "%d pairs of contour lines at different elevations cross each "
            "other. The contour data is topologically invalid and any surface "
            "built from it will be wrong." % crossings)
    if self_int:
        src.add_issue(
            QualityIssue.SEVERITY_WARNING,
            "%d contour line(s) are self-intersecting or otherwise invalid."
            % self_int)
    if pairs_checked > max_pairs:
        src.add_issue(
            QualityIssue.SEVERITY_INFO,
            "Contour crossing check was truncated after %d candidate pairs."
            % max_pairs)
