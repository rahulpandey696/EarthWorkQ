# -*- coding: utf-8 -*-
"""
Grid construction for EarthWorkQ.

Everything downstream of this module works on aligned numpy arrays:

    ground   (H, W)  existing surface, NaN outside the interpolation domain
    design   (H, W)  formation surface
    weight   (H, W)  fraction of each cell that lies inside the area of interest

The fractional weight is what makes boundary cells honest.  A simple
"is the cell centre inside the polygon" test introduces an error proportional
to the perimeter of the site, which for a narrow strip - a road corridor, a
channel - is not a rounding difference.  Coverage is obtained by rasterising
the area of interest at a sub-cell resolution and averaging each block back
down.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsVectorLayer,
)

from . import compat
from . import interpolation as interp
from .sources import RasterSource, ScatterSource, SourceError

NODATA = -9999.0
FID_FIELD = "_ewq_fid"


class GridError(RuntimeError):
    pass


# ----------------------------------------------------------------------
class TargetGrid(object):
    """A regular grid: origin at the top-left corner, row 0 at the top."""

    def __init__(self, xmin, ymin, xmax, ymax, cell_size, crs):
        self.cell = float(cell_size)
        if self.cell <= 0:
            raise GridError("Cell size must be greater than zero.")
        self.ncols = max(1, int(math.ceil((xmax - xmin) / self.cell)))
        self.nrows = max(1, int(math.ceil((ymax - ymin) / self.cell)))
        self.xmin = float(xmin)
        self.ymax = float(ymin) + self.nrows * self.cell
        self.xmax = self.xmin + self.ncols * self.cell
        self.ymin = float(ymin)
        self.crs = crs

    # -- geometry ------------------------------------------------------
    @property
    def cell_area(self):
        return self.cell * self.cell

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    def extent(self):
        return QgsRectangle(self.xmin, self.ymin, self.xmax, self.ymax)

    def extent_string(self):
        return "%f,%f,%f,%f [%s]" % (self.xmin, self.xmax, self.ymin,
                                     self.ymax, self.crs.authid())

    def geotransform(self):
        return (self.xmin, self.cell, 0.0, self.ymax, 0.0, -self.cell)

    def cell_centres(self):
        """Coordinate arrays of cell centres, shaped like the grid."""
        xs = self.xmin + (np.arange(self.ncols) + 0.5) * self.cell
        ys = self.ymax - (np.arange(self.nrows) + 0.5) * self.cell
        return np.meshgrid(xs, ys)

    def refined(self, factor):
        return TargetGrid(self.xmin, self.ymin, self.xmax, self.ymax,
                          self.cell / float(factor), self.crs)

    def describe(self):
        return [
            ("Grid cell size", "%.3f" % self.cell),
            ("Grid size", "%d columns x %d rows (%s cells)"
             % (self.ncols, self.nrows, "{:,}".format(self.ncols * self.nrows))),
            ("Grid origin", "%.3f, %.3f" % (self.xmin, self.ymax)),
        ]


def recommend_cell_size(source, aoi_extent=None):
    """Cell size should follow the data, not the user's optimism.

    For scattered points a cell of about half the mean nearest-neighbour
    spacing is the practical limit; anything finer invents detail the survey
    never captured.  For a DEM the native pixel is the answer.
    """
    if isinstance(source, RasterSource):
        cx, cy = source.native_cell_size()
        return round(max(cx, cy), 4)
    spacing = source.mean_point_spacing()
    cell = spacing / 2.0
    # round to a tidy number
    if cell <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(cell))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if cell <= m * mag:
            return round(m * mag, 4)
    return round(cell, 4)


def build_grid(source, aoi_layer, cell_size=None, buffer_cells=1):
    """Grid covering the area of interest, snapped outward by a cell or two."""
    e = aoi_layer.extent()
    if cell_size is None:
        cell_size = recommend_cell_size(source)
    pad = buffer_cells * cell_size
    return TargetGrid(e.xMinimum() - pad, e.yMinimum() - pad,
                      e.xMaximum() + pad, e.yMaximum() + pad,
                      cell_size, aoi_layer.crs())


# ----------------------------------------------------------------------
# area of interest coverage
# ----------------------------------------------------------------------
def _aoi_with_fid(aoi_layer, selected_only=False):
    """Memory copy of the area of interest carrying a dense integer id."""
    crs = aoi_layer.crs().authid()
    mem = QgsVectorLayer("Polygon?crs=%s" % crs, "ewq_aoi", "memory")
    dp = mem.dataProvider()
    dp.addAttributes([compat.make_field(FID_FIELD, "int")])
    mem.updateFields()

    feats = []
    mapping = {}
    iterator = (aoi_layer.getSelectedFeatures() if selected_only
                else aoi_layer.getFeatures())
    for i, f in enumerate(iterator, start=1):
        geom = f.geometry()
        if geom is None or geom.isEmpty():
            continue
        if not geom.isGeosValid():
            fixed = geom.makeValid()
            if fixed is not None and not fixed.isEmpty():
                geom = fixed
        nf = QgsFeature(mem.fields())
        nf.setGeometry(geom)
        nf.setAttribute(FID_FIELD, i)
        feats.append(nf)
        mapping[i] = f.id()
    if not feats:
        raise GridError(
            "The area of interest layer contains no usable polygons."
            + (" No features are selected." if selected_only else ""))
    dp.addFeatures(feats)
    mem.updateExtents()
    return mem, mapping


def coverage_weights(grid, aoi_layer, subsample=8, selected_only=False,
                     feedback=None):
    """Fraction of each grid cell inside each area-of-interest polygon.

    Returns (weights_by_id, mapping, combined) where weights_by_id maps the
    dense id to an (H, W) float array in [0, 1].
    """
    import processing
    from osgeo import gdal

    mem, mapping = _aoi_with_fid(aoi_layer, selected_only)
    fine = grid.refined(subsample)
    tmp = os.path.join(tempfile.gettempdir(),
                       "ewq_cover_%d.tif" % abs(hash(grid.extent_string())))
    processing.run("gdal:rasterize", {
        "INPUT": mem,
        "FIELD": FID_FIELD,
        "BURN": 0,
        "USE_Z": False,
        "UNITS": 1,
        "WIDTH": fine.cell,
        "HEIGHT": fine.cell,
        "EXTENT": "%f,%f,%f,%f" % (fine.xmin, fine.xmax, fine.ymin, fine.ymax),
        "NODATA": 0,
        "OPTIONS": "COMPRESS=LZW",
        "DATA_TYPE": 4,          # UInt32
        "INIT": 0,
        "INVERT": False,
        "EXTRA": "",
        "OUTPUT": tmp,
    })

    ds = gdal.Open(tmp)
    if ds is None:
        raise GridError("Failed to rasterise the area of interest.")
    fine_arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # trim / pad to an exact multiple of the subsample factor
    h, w = grid.nrows * subsample, grid.ncols * subsample
    a = np.zeros((h, w), dtype=fine_arr.dtype)
    hh = min(h, fine_arr.shape[0])
    ww = min(w, fine_arr.shape[1])
    a[:hh, :ww] = fine_arr[:hh, :ww]

    weights = {}
    blocks = a.reshape(grid.nrows, subsample, grid.ncols, subsample)
    for fid in mapping:
        weights[fid] = (blocks == fid).mean(axis=(1, 3)).astype(float)
    combined = np.clip(sum(weights.values()), 0.0, 1.0) if weights else \
        np.zeros(grid.shape)

    try:
        os.remove(tmp)
    except OSError:
        pass
    return weights, mapping, combined


# ----------------------------------------------------------------------
# gridding an elevation source
# ----------------------------------------------------------------------
def grid_from_source(source, grid, engine, feedback=None, clip_to_domain=True):
    """Evaluate an elevation source onto the target grid.

    Returns (array, domain_geometry).  Cells outside the interpolation domain
    are NaN - the engines never extrapolate.
    """
    if isinstance(source, RasterSource):
        return _grid_from_raster(source, grid), None

    if getattr(engine, "produces_raster", False):
        arr = _grid_via_processing(source, grid, engine, feedback)
    else:
        gx, gy = grid.cell_centres()
        engine.fit(source.x, source.y, source.z)
        arr = np.asarray(engine.evaluate(gx, gy), dtype=float)
        arr = arr.reshape(grid.shape)

    domain = None
    if clip_to_domain:
        domain = source.convex_hull()
        mask = _geometry_mask(grid, domain)
        arr = np.where(mask, arr, np.nan)
    arr[~np.isfinite(arr)] = np.nan
    return arr, domain


def _grid_from_raster(source, grid):
    """Warp a DEM onto the target grid, nearest for integers, bilinear else."""
    import processing
    from osgeo import gdal

    out = os.path.join(tempfile.gettempdir(),
                       "ewq_dem_%d.tif" % abs(hash(source.name + grid.extent_string())))
    processing.run("gdal:warpreproject", {
        "INPUT": source.source_path(),
        "SOURCE_CRS": source.crs.authid(),
        "TARGET_CRS": grid.crs.authid(),
        "RESAMPLING": 1,           # bilinear
        "NODATA": NODATA,
        "TARGET_RESOLUTION": grid.cell,
        "OPTIONS": "COMPRESS=LZW|PREDICTOR=3",
        "DATA_TYPE": 6,            # Float32
        "TARGET_EXTENT": "%f,%f,%f,%f" % (grid.xmin, grid.xmax, grid.ymin, grid.ymax),
        "TARGET_EXTENT_CRS": grid.crs.authid(),
        "MULTITHREADING": False,
        "EXTRA": "",
        "OUTPUT": out,
    })
    ds = gdal.Open(out)
    if ds is None:
        raise GridError("Could not resample the DEM '%s' onto the working grid."
                        % source.name)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    ds = None
    arr[np.isclose(arr, NODATA)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    if arr.shape != grid.shape:
        fixed = np.full(grid.shape, np.nan)
        hh = min(arr.shape[0], grid.nrows)
        ww = min(arr.shape[1], grid.ncols)
        fixed[:hh, :ww] = arr[:hh, :ww]
        arr = fixed
    try:
        os.remove(out)
    except OSError:
        pass
    return arr


def _points_to_memory_layer(source, grid):
    crs = grid.crs.authid()
    mem = QgsVectorLayer("Point?crs=%s" % crs, "ewq_pts", "memory")
    dp = mem.dataProvider()
    dp.addAttributes([compat.make_field("z", "double")])
    mem.updateFields()
    feats = []
    for xv, yv, zv in zip(source.x, source.y, source.z):
        f = QgsFeature(mem.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(xv), float(yv))))
        f.setAttribute("z", float(zv))
        feats.append(f)
    dp.addFeatures(feats)
    mem.updateExtents()
    return mem


def _grid_via_processing(source, grid, engine, feedback=None):
    """Run a Processing-backed engine (GRASS or native QGIS TIN)."""
    import processing
    from osgeo import gdal

    out = os.path.join(tempfile.gettempdir(),
                       "ewq_interp_%d.tif" % abs(hash(engine.id + grid.extent_string())))

    if engine.id == "qgis_tin":
        layer = _points_to_memory_layer(source, grid)
        # data string: layer_source::~::layer_type::~::attribute_index::~::input_type
        data = "%s::~::0::~::0::~::0" % layer.id()
        QgsProject.instance().addMapLayer(layer, False)
        try:
            processing.run("qgis:tininterpolation", {
                "INTERPOLATION_DATA": data,
                "METHOD": 0,
                "EXTENT": "%f,%f,%f,%f" % (grid.xmin, grid.xmax, grid.ymin, grid.ymax),
                "PIXEL_SIZE": grid.cell,
                "OUTPUT": out,
            })
        finally:
            QgsProject.instance().removeMapLayer(layer.id())
    elif engine.id == "grass_contour":
        # r.surf.contour needs a raster of the contour lines
        lines = _contour_lines_layer(source, grid)
        rast = os.path.join(tempfile.gettempdir(), "ewq_cont_%d.tif" % abs(hash(out)))
        processing.run("gdal:rasterize", {
            "INPUT": lines, "FIELD": "z", "BURN": 0, "USE_Z": False, "UNITS": 1,
            "WIDTH": grid.cell, "HEIGHT": grid.cell,
            "EXTENT": "%f,%f,%f,%f" % (grid.xmin, grid.xmax, grid.ymin, grid.ymax),
            "NODATA": NODATA, "OPTIONS": "COMPRESS=LZW", "DATA_TYPE": 6,
            "INIT": None, "INVERT": False, "EXTRA": "", "OUTPUT": rast,
        })
        processing.run("grass7:r.surf.contour", {
            "input": rast,
            "GRASS_REGION_PARAMETER": "%f,%f,%f,%f" % (grid.xmin, grid.xmax,
                                                       grid.ymin, grid.ymax),
            "GRASS_REGION_CELLSIZE_PARAMETER": grid.cell,
            "output": out,
        })
        try:
            os.remove(rast)
        except OSError:
            pass
    elif engine.id == "grass_rst":
        layer = _points_to_memory_layer(source, grid)
        processing.run("grass7:v.surf.rst", {
            "input": layer,
            "zcolumn": "z",
            "tension": float(engine.params.get("tension", 40.0)),
            "smooth": float(engine.params.get("smooth", 0.1)),
            "GRASS_REGION_PARAMETER": "%f,%f,%f,%f" % (grid.xmin, grid.xmax,
                                                       grid.ymin, grid.ymax),
            "GRASS_REGION_CELLSIZE_PARAMETER": grid.cell,
            "elevation": out,
        })
    else:
        raise GridError("Engine '%s' cannot be run through Processing."
                        % engine.id)

    ds = gdal.Open(out)
    if ds is None:
        raise GridError(
            "The %s engine did not produce a raster. Check that its provider "
            "is enabled in Processing > Options." % engine.display_name)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nd is not None:
        arr[np.isclose(arr, nd)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    if arr.shape != grid.shape:
        fixed = np.full(grid.shape, np.nan)
        hh, ww = min(arr.shape[0], grid.nrows), min(arr.shape[1], grid.ncols)
        fixed[:hh, :ww] = arr[:hh, :ww]
        arr = fixed
    try:
        os.remove(out)
    except OSError:
        pass
    return arr


def _contour_lines_layer(source, grid):
    """Rebuild a line layer from a ContourSource for the GRASS route."""
    crs = grid.crs.authid()
    mem = QgsVectorLayer("LineString?crs=%s" % crs, "ewq_contours", "memory")
    dp = mem.dataProvider()
    dp.addAttributes([compat.make_field("z", "double")])
    mem.updateFields()
    feats = []
    for geom, elev in getattr(source, "_geometries", []):
        f = QgsFeature(mem.fields())
        f.setGeometry(QgsGeometry(geom))
        f.setAttribute("z", float(elev))
        feats.append(f)
    if not feats:
        raise GridError("The contour source carries no line geometries.")
    dp.addFeatures(feats)
    mem.updateExtents()
    return mem


def _geometry_mask(grid, geometry):
    """Boolean mask of cells whose centre lies inside a geometry."""
    import processing
    from osgeo import gdal
    from qgis.core import QgsFeature as _F

    mem = QgsVectorLayer("Polygon?crs=%s" % grid.crs.authid(), "ewq_dom", "memory")
    dp = mem.dataProvider()
    f = _F()
    f.setGeometry(QgsGeometry(geometry))
    dp.addFeatures([f])
    mem.updateExtents()

    out = os.path.join(tempfile.gettempdir(),
                       "ewq_domain_%d.tif" % abs(hash(grid.extent_string())))
    processing.run("gdal:rasterize", {
        "INPUT": mem, "FIELD": None, "BURN": 1, "USE_Z": False, "UNITS": 1,
        "WIDTH": grid.cell, "HEIGHT": grid.cell,
        "EXTENT": "%f,%f,%f,%f" % (grid.xmin, grid.xmax, grid.ymin, grid.ymax),
        "NODATA": 0, "OPTIONS": "COMPRESS=LZW", "DATA_TYPE": 0,
        "INIT": 0, "INVERT": False, "EXTRA": "", "OUTPUT": out,
    })
    ds = gdal.Open(out)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    mask = np.zeros(grid.shape, dtype=bool)
    hh, ww = min(arr.shape[0], grid.nrows), min(arr.shape[1], grid.ncols)
    mask[:hh, :ww] = arr[:hh, :ww] > 0
    try:
        os.remove(out)
    except OSError:
        pass
    return mask


def write_raster(path, array, grid, nodata=NODATA, dtype=None):
    """Write an aligned numpy array to a compressed GeoTIFF."""
    from osgeo import gdal, osr
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    data = np.array(array, dtype=float, copy=True)
    data[~np.isfinite(data)] = nodata
    ds = driver.Create(path, grid.ncols, grid.nrows, 1,
                       dtype or gdal.GDT_Float32,
                       options=["COMPRESS=LZW", "PREDICTOR=3", "TILED=YES"])
    ds.SetGeoTransform(grid.geotransform())
    srs = osr.SpatialReference()
    srs.ImportFromWkt(grid.crs.toWkt())
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    ds = None
    return path
