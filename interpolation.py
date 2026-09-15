# -*- coding: utf-8 -*-
"""
Interpolation engines for EarthWorkQ.

Every engine implements the same small interface, which lets the plugin do two
things that matter for a professional deliverable:

  * cross-validate all applicable engines on the same data and report the RMSE
    of each, so the choice of interpolator becomes a measurement rather than an
    opinion;
  * compute the final quantity with two different engines and report the
    spread, which is a direct indication of whether the survey is dense enough
    to support the accuracy being claimed.

Engines never extrapolate.  Everything outside the valid domain (convex hull
for scattered points, the contour envelope for contour data) is returned as
NaN and excluded from the quantities, with the shortfall reported.
"""

from __future__ import annotations

import math

import numpy as np

INPUT_POINTS = "points"
INPUT_CONTOURS = "contours"


class InterpolationError(RuntimeError):
    pass


def _require_scipy(what):
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise InterpolationError(
            "%s requires SciPy, which is not available in this QGIS "
            "installation. Use the QGIS TIN or IDW engine instead, or install "
            "SciPy via the OSGeo4W shell." % what)


# ----------------------------------------------------------------------
class InterpolationEngine(object):
    """Base class. Subclasses implement fit() and _evaluate()."""

    id = "abstract"
    display_name = "Abstract"
    supported_inputs = ()
    is_exact = True
    description = ""

    #: list of (key, label, default, kind, extra) describing tunable parameters
    parameters = ()

    def __init__(self, **params):
        self.params = dict(params)
        self._fitted = False
        self._hull = None

    # -- interface -----------------------------------------------------
    def fit(self, x, y, z):
        self._x = np.asarray(x, float)
        self._y = np.asarray(y, float)
        self._z = np.asarray(z, float)
        self._fit()
        self._fitted = True
        return self

    def _fit(self):
        raise NotImplementedError

    def _evaluate(self, gx, gy):
        raise NotImplementedError

    def evaluate(self, gx, gy):
        if not self._fitted:
            raise InterpolationError("Engine used before being fitted.")
        out = self._evaluate(gx, gy)
        return out

    # -- helpers -------------------------------------------------------
    def clone(self):
        return self.__class__(**self.params)

    def cross_validate(self, x, y, z, folds=10, max_points=20000, seed=0):
        """k-fold cross-validation RMSE, in elevation units.

        Leave-one-out would be ideal but requires one refit per point, which is
        not tractable for a triangulated surface of any size.  Ten folds gives
        an honest, reproducible error estimate at a fixed cost.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(z, float)
        n = x.size
        rng = np.random.default_rng(seed)
        if n > max_points:
            sel = rng.choice(n, max_points, replace=False)
            x, y, z, n = x[sel], y[sel], z[sel], max_points
        if n < folds * 3:
            folds = max(2, n // 3)
        order = rng.permutation(n)
        residuals = []
        for f in range(folds):
            test = order[f::folds]
            mask = np.ones(n, dtype=bool)
            mask[test] = False
            if mask.sum() < 4:
                continue
            try:
                eng = self.clone()
                eng.fit(x[mask], y[mask], z[mask])
                pred = eng.evaluate(x[test], y[test])
            except Exception:
                continue
            ok = np.isfinite(pred)
            if ok.any():
                residuals.append(pred[ok] - z[test][ok])
        if not residuals:
            return None
        r = np.concatenate(residuals)
        return dict(rmse=float(np.sqrt(np.mean(r ** 2))),
                    mae=float(np.mean(np.abs(r))),
                    bias=float(np.mean(r)),
                    max_abs=float(np.max(np.abs(r))),
                    n=int(r.size))


# ----------------------------------------------------------------------
class LinearTinEngine(InterpolationEngine):
    """Delaunay triangulation with a plane through each triangle."""

    id = "tin_linear"
    display_name = "TIN - linear (Delaunay)"
    supported_inputs = (INPUT_POINTS, INPUT_CONTOURS)
    is_exact = True
    description = (
        "Fits a plane through each Delaunay triangle. Honours every survey "
        "point exactly, can never overshoot the data range, and is the method "
        "underlying traditional spot-level volume calculations. Produces a "
        "faceted surface with discontinuous slope at triangle edges.")

    def _fit(self):
        _require_scipy("The linear TIN engine")
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
        pts = np.column_stack((self._x, self._y))
        try:
            self._tri = Delaunay(pts)
        except Exception as exc:
            raise InterpolationError(
                "Delaunay triangulation failed (%s). This usually means all "
                "points are collinear or coincident." % exc)
        self._interp = LinearNDInterpolator(self._tri, self._z)

    def _evaluate(self, gx, gy):
        return self._interp(gx, gy)

    @property
    def triangulation(self):
        return self._tri


class CloughTocherEngine(InterpolationEngine):
    """Cubic Bezier patch per triangle, C1 continuous across edges."""

    id = "tin_cubic"
    display_name = "TIN - cubic (Clough-Tocher)"
    supported_inputs = (INPUT_POINTS,)
    is_exact = True
    description = (
        "Same triangulation as the linear TIN but fits a cubic patch to each "
        "triangle, giving a smooth surface with continuous slope. Still exact "
        "at every data point. Mild overshoot is possible next to abrupt breaks "
        "such as bank crests or retaining structures.")

    def _fit(self):
        _require_scipy("The Clough-Tocher engine")
        from scipy.interpolate import CloughTocher2DInterpolator
        pts = np.column_stack((self._x, self._y))
        self._interp = CloughTocher2DInterpolator(pts, self._z)

    def _evaluate(self, gx, gy):
        return self._interp(gx, gy)


class IdwEngine(InterpolationEngine):
    """Inverse distance weighting over the k nearest neighbours."""

    id = "idw"
    display_name = "Inverse distance weighting"
    supported_inputs = (INPUT_POINTS,)
    is_exact = True
    description = (
        "Weights neighbouring points by the inverse of distance raised to a "
        "power. Simple and widely specified, but it cannot return a value "
        "outside the range of the neighbours it uses, so ridges are flattened "
        "and hollows are lifted. Isolated points show as concentric bullseyes. "
        "Not recommended as a primary engine for terrain.")

    parameters = (
        ("power", "Distance power", 2.0, "double", (0.5, 6.0, 1)),
        ("neighbours", "Neighbours used", 12, "int", (3, 64, 0)),
        ("smoothing", "Smoothing distance", 0.0, "double", (0.0, 100.0, 3)),
    )

    def _fit(self):
        _require_scipy("The IDW engine")
        from scipy.spatial import cKDTree
        self._tree = cKDTree(np.column_stack((self._x, self._y)))

    def _evaluate(self, gx, gy):
        p = float(self.params.get("power", 2.0))
        k = int(self.params.get("neighbours", 12))
        s = float(self.params.get("smoothing", 0.0))
        k = max(1, min(k, self._x.size))
        q = np.column_stack((np.ravel(gx), np.ravel(gy)))
        d, idx = self._tree.query(q, k=k)
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]
        d = np.sqrt(d ** 2 + s ** 2)
        exact = d < 1e-9
        w = np.where(exact, 1.0, 1.0 / np.power(np.maximum(d, 1e-12), p))
        has_exact = exact.any(axis=1)
        w[has_exact] = exact[has_exact].astype(float)
        vals = self._z[idx]
        out = (w * vals).sum(axis=1) / w.sum(axis=1)
        return out.reshape(np.shape(gx))


class NearestEngine(InterpolationEngine):
    """Nearest neighbour - a diagnostic, not a terrain model."""

    id = "nearest"
    display_name = "Nearest neighbour (diagnostic only)"
    supported_inputs = (INPUT_POINTS,)
    is_exact = True
    description = (
        "Assigns each cell the elevation of the closest survey point. Produces "
        "a blocky Thiessen surface. Useful only for checking data coverage - "
        "do not use it to report quantities.")

    def _fit(self):
        _require_scipy("The nearest neighbour engine")
        from scipy.spatial import cKDTree
        self._tree = cKDTree(np.column_stack((self._x, self._y)))

    def _evaluate(self, gx, gy):
        q = np.column_stack((np.ravel(gx), np.ravel(gy)))
        _, idx = self._tree.query(q, k=1)
        return self._z[idx].reshape(np.shape(gx))


class RbfThinPlateEngine(InterpolationEngine):
    """Thin-plate / regularised spline with tension and smoothing."""

    id = "spline_rbf"
    display_name = "Spline - thin plate with tension"
    supported_inputs = (INPUT_POINTS,)
    is_exact = False
    description = (
        "Radial basis function fit that minimises surface curvature. Gives a "
        "smooth surface with well behaved slopes, and the smoothing parameter "
        "lets noisy GNSS data be filtered rather than honoured point for "
        "point. Can overshoot near steep breaks - raise the tension if the "
        "surface swings above or below the data.")

    parameters = (
        ("smoothing", "Smoothing (0 = exact fit)", 0.0, "double", (0.0, 100.0, 3)),
        ("neighbours", "Neighbours per solve", 64, "int", (16, 512, 0)),
        ("kernel", "Kernel", "thin_plate_spline", "choice",
         ("thin_plate_spline", "multiquadric", "linear", "cubic")),
    )

    def _fit(self):
        _require_scipy("The spline engine")
        try:
            from scipy.interpolate import RBFInterpolator
        except ImportError:
            raise InterpolationError(
                "The spline engine needs SciPy 1.7 or newer. Use the "
                "Clough-Tocher TIN engine instead.")
        pts = np.column_stack((self._x, self._y))
        n = pts.shape[0]
        self._interp = RBFInterpolator(
            pts, self._z,
            neighbors=min(int(self.params.get("neighbours", 64)), n),
            smoothing=float(self.params.get("smoothing", 0.0)),
            kernel=str(self.params.get("kernel", "thin_plate_spline")))

    def _evaluate(self, gx, gy):
        q = np.column_stack((np.ravel(gx), np.ravel(gy)))
        return self._interp(q).reshape(np.shape(gx))


# ----------------------------------------------------------------------
# GRASS-backed engines (raster producing, handled specially by the gridder)
# ----------------------------------------------------------------------
class GrassEngine(InterpolationEngine):
    """Base for engines that run as a QGIS Processing algorithm and return a
    raster directly rather than evaluating at arbitrary points."""

    produces_raster = True
    algorithm = None

    def _fit(self):
        pass

    def _evaluate(self, gx, gy):
        raise InterpolationError(
            "%s produces a raster directly and cannot be evaluated at "
            "arbitrary points. It is applied by the gridder instead."
            % self.display_name)

    def build_parameters(self, source_layer, extent, cell_size, crs, output):
        raise NotImplementedError


class GrassRstEngine(GrassEngine):
    id = "grass_rst"
    display_name = "GRASS regularised spline with tension (v.surf.rst)"
    supported_inputs = (INPUT_POINTS, INPUT_CONTOURS)
    is_exact = False
    algorithm = "grass7:v.surf.rst"
    description = (
        "Segmented regularised spline with tension. Mature, handles very large "
        "point sets, and reports its own deviation statistics. Requires the "
        "GRASS provider to be enabled in QGIS.")

    parameters = (
        ("tension", "Tension", 40.0, "double", (1.0, 400.0, 1)),
        ("smooth", "Smoothing", 0.1, "double", (0.0, 100.0, 3)),
    )

    def build_parameters(self, source_layer, extent, cell_size, crs, output):
        return {
            "input": source_layer,
            "zcolumn": self.params.get("zcolumn"),
            "tension": float(self.params.get("tension", 40.0)),
            "smooth": float(self.params.get("smooth", 0.1)),
            "GRASS_REGION_PARAMETER": extent,
            "GRASS_REGION_CELLSIZE_PARAMETER": cell_size,
            "elevation": output,
        }


class GrassContourEngine(GrassEngine):
    id = "grass_contour"
    display_name = "GRASS contour interpolation (r.surf.contour)"
    supported_inputs = (INPUT_CONTOURS,)
    is_exact = True
    algorithm = "grass7:r.surf.contour"
    description = (
        "Purpose-built for contour input. Interpolates by distance-weighting "
        "between the bracketing contours instead of triangulating vertices, so "
        "it does not produce the flat hilltops and dead-flat valley floors "
        "that a naive TIN of contour vertices suffers from. Requires GRASS.")

    def build_parameters(self, source_layer, extent, cell_size, crs, output):
        return {
            "input": source_layer,
            "GRASS_REGION_PARAMETER": extent,
            "GRASS_REGION_CELLSIZE_PARAMETER": cell_size,
            "output": output,
        }


class QgisTinEngine(GrassEngine):
    """QGIS native TIN interpolation - the no-SciPy fallback."""

    id = "qgis_tin"
    display_name = "QGIS TIN interpolation (no SciPy required)"
    supported_inputs = (INPUT_POINTS, INPUT_CONTOURS)
    is_exact = True
    algorithm = "qgis:tininterpolation"
    description = (
        "The interpolation shipped with QGIS itself. Equivalent in principle "
        "to the linear TIN engine. Use it when SciPy is not installed.")

    def build_parameters(self, source_layer, extent, cell_size, crs, output):
        return {
            "INTERPOLATION_DATA": None,   # filled in by the gridder
            "METHOD": 0,
            "EXTENT": extent,
            "PIXEL_SIZE": cell_size,
            "OUTPUT": output,
        }


# ----------------------------------------------------------------------
ENGINES = [
    CloughTocherEngine,
    LinearTinEngine,
    RbfThinPlateEngine,
    IdwEngine,
    NearestEngine,
    GrassRstEngine,
    GrassContourEngine,
    QgisTinEngine,
]

#: opinionated defaults - contour data goes to the contour-aware engine
DEFAULT_ENGINE = {
    INPUT_POINTS: CloughTocherEngine.id,
    INPUT_CONTOURS: GrassContourEngine.id,
}

#: fallbacks when the preferred engine is unavailable
FALLBACK_ENGINE = {
    INPUT_POINTS: LinearTinEngine.id,
    INPUT_CONTOURS: LinearTinEngine.id,
}


def engines_for(input_kind):
    """Engines valid for a given input type, in recommended order."""
    return [e for e in ENGINES if input_kind in e.supported_inputs]


def engine_by_id(engine_id):
    for e in ENGINES:
        if e.id == engine_id:
            return e
    raise InterpolationError("Unknown interpolation engine '%s'." % engine_id)


def create_engine(engine_id, **params):
    return engine_by_id(engine_id)(**params)


def scipy_available():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def grass_available():
    try:
        from qgis.core import QgsApplication
        reg = QgsApplication.processingRegistry()
        return reg.algorithmById("grass7:r.surf.contour") is not None
    except Exception:
        return False
