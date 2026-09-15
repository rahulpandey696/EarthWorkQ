# -*- coding: utf-8 -*-
"""
Design (formation) surfaces for EarthWorkQ.

Three kinds, all reduced to an array on the working grid:

    FlatLevel     a single formation level
    SlopedPlane   a spot level plus grades in X and Y - real platforms drain
    DesignRaster  a full design DEM produced elsewhere
"""

from __future__ import annotations

import numpy as np


class DesignSurface(object):
    kind = "abstract"
    is_planar = False        # true if linear over any triangle - enables exact TIN volumes
    is_level_adjustable = False   # true if a balance sweep can shift it vertically

    def to_grid(self, grid):
        raise NotImplementedError

    def shifted(self, delta):
        """Return a copy raised by delta - used by the balance solver."""
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError


class FlatLevel(DesignSurface):
    kind = "flat"
    is_planar = True
    is_level_adjustable = True

    def __init__(self, level):
        self.level = float(level)

    def to_grid(self, grid):
        return np.full(grid.shape, self.level, dtype=float)

    def shifted(self, delta):
        return FlatLevel(self.level + delta)

    def describe(self):
        return [("Design surface", "Flat formation level"),
                ("Formation level", "%.3f" % self.level)]


class SlopedPlane(DesignSurface):
    """z = z0 + gx * (x - x0) + gy * (y - y0), grades as decimals (0.01 = 1%)."""

    kind = "plane"
    is_planar = True
    is_level_adjustable = True

    def __init__(self, z0, x0, y0, grade_x=0.0, grade_y=0.0):
        self.z0 = float(z0)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.grade_x = float(grade_x)
        self.grade_y = float(grade_y)

    def to_grid(self, grid):
        gx, gy = grid.cell_centres()
        return self.z0 + self.grade_x * (gx - self.x0) + self.grade_y * (gy - self.y0)

    def evaluate(self, x, y):
        return self.z0 + self.grade_x * (np.asarray(x) - self.x0) \
            + self.grade_y * (np.asarray(y) - self.y0)

    def shifted(self, delta):
        return SlopedPlane(self.z0 + delta, self.x0, self.y0,
                           self.grade_x, self.grade_y)

    def describe(self):
        return [
            ("Design surface", "Sloped formation plane"),
            ("Level at control point", "%.3f" % self.z0),
            ("Control point", "%.2f, %.2f" % (self.x0, self.y0)),
            ("Grade in X", "%.3f %% (%s)" % (self.grade_x * 100,
                                             "1 in %.0f" % (1 / self.grade_x)
                                             if self.grade_x else "level")),
            ("Grade in Y", "%.3f %% (%s)" % (self.grade_y * 100,
                                             "1 in %.0f" % (1 / self.grade_y)
                                             if self.grade_y else "level")),
        ]


class DesignRaster(DesignSurface):
    kind = "raster"
    is_planar = False
    is_level_adjustable = True   # can still be raised or lowered bodily

    def __init__(self, source, offset=0.0):
        self.source = source           # a RasterSource
        self.offset = float(offset)
        self._cache = {}

    def to_grid(self, grid):
        from .grid import _grid_from_raster   # imported late: needs QGIS
        key = id(grid)
        if key not in self._cache:
            self._cache[key] = _grid_from_raster(self.source, grid)
        return self._cache[key] + self.offset

    def shifted(self, delta):
        d = DesignRaster(self.source, self.offset + delta)
        d._cache = self._cache
        return d

    def describe(self):
        rows = [("Design surface", "Design DEM"),
                ("Design DEM", self.source.name)]
        if abs(self.offset) > 1e-9:
            rows.append(("Applied vertical offset", "%+.3f" % self.offset))
        return rows
