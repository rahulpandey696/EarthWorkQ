# -*- coding: utf-8 -*-
"""
The EarthWorkQ analysis pipeline.

One object, ``EarthworkAnalysis``, holds the settings; ``run()`` executes the
pipeline and returns an ``AnalysisResult`` that the GUI, the Processing
algorithms and the report writer all consume.  Keeping the whole computation
here means the interface can be replaced without touching the engineering.
"""

from __future__ import annotations

import math
import os

import numpy as np

from . import balance as balance_mod
from . import interpolation as interp
from . import outputs as out_mod
from . import volume as vol_mod
from .design import DesignRaster, DesignSurface, FlatLevel, SlopedPlane
from .grid import (
    build_grid,
    coverage_weights,
    grid_from_source,
    recommend_cell_size,
)
from .sources import QualityIssue, RasterSource, ScatterSource, SourceError


class AnalysisError(RuntimeError):
    pass


class NullFeedback(object):
    def setProgress(self, value):
        pass

    def pushInfo(self, text):
        pass

    def isCanceled(self):
        return False


class AnalysisResult(object):
    def __init__(self):
        self.grid = None
        self.ground = None
        self.ground_stripped = None
        self.design = None
        self.design_grid = None
        self.depth = None
        self.weight = None
        self.weights_by_id = {}
        self.mapping = {}
        self.total = None
        self.per_feature = {}
        self.topsoil_volume = 0.0
        self.curve = None
        self.balance_level = None
        self.minimum_movement_level = None
        self.cross_validation = {}
        self.engine_comparison = {}
        self.tin_check = None
        self.convergence = None
        self.volume_uncertainty = None
        self.issues = []
        self.written = {}
        self.settings_summary = []
        self.unit = "m"

    def add_issue(self, severity, message):
        self.issues.append(QualityIssue(severity, message))

    @property
    def has_errors(self):
        return any(i.severity == QualityIssue.SEVERITY_ERROR for i in self.issues)


class EarthworkAnalysis(object):
    """Settings container and pipeline driver."""

    def __init__(self, source, aoi_layer, design, soil=None,
                 engine_id=None, engine_params=None, cell_size=None,
                 subsample=8, selected_only=False,
                 run_cross_validation=True, run_convergence=True,
                 run_engine_comparison=False, comparison_engine_id=None,
                 balance_steps=200, zone_tolerance=0.05,
                 contour_interval=None, output_dir=None, prefix="earthworkq",
                 write_rasters=True, write_zones=True, write_contours=False,
                 write_chart=True, write_csv=True, write_back_attributes=True,
                 add_to_project=True, unit="m", solve_balance_first=False):
        self.source = source
        self.aoi_layer = aoi_layer
        self.design = design
        self.soil = soil
        self.engine_id = engine_id
        self.engine_params = engine_params or {}
        self.cell_size = cell_size
        self.subsample = int(subsample)
        self.selected_only = bool(selected_only)
        self.run_cross_validation = run_cross_validation
        self.run_convergence = run_convergence
        self.run_engine_comparison = run_engine_comparison
        self.comparison_engine_id = comparison_engine_id
        self.balance_steps = int(balance_steps)
        self.zone_tolerance = float(zone_tolerance)
        self.contour_interval = contour_interval
        self.output_dir = output_dir
        self.prefix = prefix
        self.write_rasters = write_rasters
        self.write_zones = write_zones
        self.write_contours = write_contours
        self.write_chart = write_chart
        self.write_csv = write_csv
        self.write_back_attributes = write_back_attributes
        self.add_to_project = add_to_project
        self.unit = unit
        #: replace the stated formation level with the solved balancing level
        self.solve_balance_first = bool(solve_balance_first)

    # ------------------------------------------------------------------
    def _path(self, suffix, ext):
        if not self.output_dir:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, "%s_%s.%s" % (self.prefix, suffix, ext))

    def _resolve_engine(self):
        kind = getattr(self.source, "kind", "points")
        if isinstance(self.source, RasterSource):
            return None
        eid = self.engine_id
        if not eid:
            eid = interp.DEFAULT_ENGINE.get(kind, interp.LinearTinEngine.id)
        cls = interp.engine_by_id(eid)
        if kind not in cls.supported_inputs:
            raise AnalysisError(
                "The %s engine cannot be used with %s input."
                % (cls.display_name, kind))
        if getattr(cls, "produces_raster", False) and cls.id.startswith("grass") \
                and not interp.grass_available():
            fallback = interp.FALLBACK_ENGINE.get(kind, interp.LinearTinEngine.id)
            cls = interp.engine_by_id(fallback)
        if not getattr(cls, "produces_raster", False) and not interp.scipy_available():
            cls = interp.engine_by_id(interp.QgisTinEngine.id)
        return cls(**self.engine_params)

    # ------------------------------------------------------------------
    def run(self, feedback=None):
        fb = feedback or NullFeedback()
        result = AnalysisResult()
        result.unit = self.unit

        for issue in getattr(self.source, "issues", []):
            result.issues.append(issue)
        if getattr(self.source, "errors", None):
            raise AnalysisError(
                "The elevation input failed validation:\n  - %s"
                % "\n  - ".join(i.message for i in self.source.errors))

        if self.aoi_layer.crs() != self.source.crs:
            result.add_issue(
                QualityIssue.SEVERITY_WARNING,
                "The area of interest (%s) and the elevation data (%s) are in "
                "different coordinate reference systems. Reproject them to a "
                "common projected CRS - results may otherwise be misaligned."
                % (self.aoi_layer.crs().authid(), self.source.crs.authid()))

        # -- grid ------------------------------------------------------
        fb.pushInfo("Building the working grid...")
        recommended = recommend_cell_size(self.source)
        cell = self.cell_size or recommended
        if cell < recommended * 0.9:
            result.add_issue(
                QualityIssue.SEVERITY_WARNING,
                "The chosen cell size of %.3f is finer than the %.3f "
                "recommended for this data density. A finer grid does not add "
                "information the survey does not contain - it only smooths the "
                "appearance of the interpolation." % (cell, recommended))
        grid = build_grid(self.source, self.aoi_layer, cell)
        result.grid = grid
        fb.setProgress(5)

        n_cells = grid.ncols * grid.nrows
        if n_cells > 40_000_000:
            raise AnalysisError(
                "The chosen cell size would produce %s cells, which is beyond "
                "what can be held in memory. Increase the cell size or reduce "
                "the area of interest." % "{:,}".format(n_cells))

        # -- coverage --------------------------------------------------
        fb.pushInfo("Computing fractional cell coverage of the area of interest...")
        weights_by_id, mapping, combined = coverage_weights(
            grid, self.aoi_layer, subsample=self.subsample,
            selected_only=self.selected_only, feedback=fb)
        result.weights_by_id = weights_by_id
        result.mapping = mapping
        result.weight = combined
        fb.setProgress(20)

        # -- existing ground ------------------------------------------
        engine = self._resolve_engine()
        if engine is not None:
            fb.pushInfo("Interpolating the existing surface with %s..."
                        % engine.display_name)
        else:
            fb.pushInfo("Resampling the existing DEM onto the working grid...")
        ground, domain = grid_from_source(self.source, grid, engine, feedback=fb)
        result.ground = ground
        fb.setProgress(45)

        covered = combined > 0
        missing = covered & ~np.isfinite(ground)
        missing_area = float(combined[missing].sum() * grid.cell_area)
        aoi_area = float(combined.sum() * grid.cell_area)
        if missing_area > 0:
            pct = missing_area / aoi_area * 100.0 if aoi_area else 0.0
            severity = (QualityIssue.SEVERITY_ERROR if pct > 5.0
                        else QualityIssue.SEVERITY_WARNING)
            result.add_issue(
                severity,
                "%.1f%% of the area of interest (%.0f %s2) lies outside the "
                "surveyed area and carries no elevation. EarthWorkQ does not "
                "extrapolate, so this area is excluded from the quantities. "
                "Either reduce the area of interest or extend the survey."
                % (pct, missing_area, self.unit))

        # -- topsoil strip --------------------------------------------
        strip = self.soil.topsoil_strip if self.soil else 0.0
        if strip > 0:
            result.topsoil_volume = float(
                np.nansum(np.where(np.isfinite(ground), combined, 0.0))
                * grid.cell_area * strip)
            ground_eff = ground - strip
        else:
            ground_eff = ground
        result.ground_stripped = ground_eff

        # -- design ----------------------------------------------------
        if self.solve_balance_first and isinstance(self.design, FlatLevel):
            fb.pushInfo("Solving for the balancing formation level...")
            sweeper = balance_mod.FlatLevelSweeper(ground_eff, combined,
                                                   grid.cell_area)
            if sweeper.has_data:
                sf0 = self.soil.shrinkage_factor if self.soil else 1.0
                solved = sweeper.solve_balance(sf0)
                result.add_issue(
                    QualityIssue.SEVERITY_INFO,
                    "The formation level was solved for balance and set to "
                    "%.3f %s (the value entered was %.3f)."
                    % (solved, self.unit, self.design.level))
                self.design = FlatLevel(solved)

        fb.pushInfo("Building the design surface...")
        design_grid = self.design.to_grid(grid)
        result.design = self.design
        result.design_grid = design_grid
        depth = vol_mod.depth_array(ground_eff, design_grid)
        result.depth = depth
        fb.setProgress(55)

        # -- quantities ------------------------------------------------
        fb.pushInfo("Integrating cut and fill quantities...")
        result.total = vol_mod.integrate(depth, combined, grid.cell_area,
                                         label="Whole site")
        result.per_feature = vol_mod.integrate_per_feature(
            depth, weights_by_id, mapping, grid.cell_area)
        fb.setProgress(65)

        # -- balance curve --------------------------------------------
        sf = self.soil.shrinkage_factor if self.soil else 1.0
        if isinstance(self.design, FlatLevel):
            curve, _ = balance_mod.sweep_levels(
                ground_eff, combined, grid.cell_area,
                n_steps=self.balance_steps, shrinkage_factor=sf)
            chosen = self.design.level
        else:
            offsets = balance_mod.auto_offset_range(
                ground_eff, design_grid, combined, n_steps=self.balance_steps)
            curve, _ = balance_mod.sweep_offsets(
                ground_eff, design_grid, combined, grid.cell_area,
                offsets, shrinkage_factor=sf)
            chosen = 0.0
        result.curve = curve
        if curve is not None:
            result.balance_level = curve.balance_level()
            result.minimum_movement_level = curve.minimum_movement_level()
            if result.balance_level is None:
                result.add_issue(
                    QualityIssue.SEVERITY_INFO,
                    "No balancing level exists within the range of ground "
                    "elevations: the site is in cut or in fill throughout, so "
                    "material must be imported or disposed of.")
        result.chosen_level = chosen
        fb.setProgress(72)

        # -- quality assurance ----------------------------------------
        if self.run_cross_validation and isinstance(self.source, ScatterSource):
            fb.pushInfo("Cross-validating the interpolation...")
            self._cross_validate(result, fb)
        fb.setProgress(80)

        if isinstance(self.source, ScatterSource) and self.design.is_planar:
            fb.pushInfo("Cross-checking against an exact TIN integration...")
            try:
                z_for_tin = self.source.z - strip
                result.tin_check = vol_mod.tin_volume(
                    self.source.x, self.source.y, z_for_tin, self.design,
                    clip_geometry=self._aoi_union())
            except Exception:
                result.tin_check = None
            self._compare_tin(result)
        fb.setProgress(86)

        if self.run_convergence:
            fb.pushInfo("Running the grid convergence check...")
            self._convergence(result, engine, fb)
        fb.setProgress(90)

        # -- outputs ---------------------------------------------------
        fb.pushInfo("Writing outputs...")
        self._write_outputs(result, fb)
        fb.setProgress(100)

        result.settings_summary = self._settings_summary(engine, cell, recommended)
        return result

    # ------------------------------------------------------------------
    def _aoi_union(self):
        from qgis.core import QgsGeometry
        geoms = []
        it = (self.aoi_layer.getSelectedFeatures() if self.selected_only
              else self.aoi_layer.getFeatures())
        for f in it:
            g = f.geometry()
            if g and not g.isEmpty():
                geoms.append(QgsGeometry(g))
        if not geoms:
            return None
        union = geoms[0]
        for g in geoms[1:]:
            union = union.combine(g)
        return union

    def _cross_validate(self, result, fb):
        kind = getattr(self.source, "kind", "points")
        candidates = [e for e in interp.engines_for(kind)
                      if not getattr(e, "produces_raster", False)]
        if not interp.scipy_available():
            return
        for cls in candidates:
            try:
                eng = cls(**(self.engine_params if cls.id == self.engine_id else {}))
                stats = eng.cross_validate(self.source.x, self.source.y,
                                           self.source.z)
                if stats:
                    result.cross_validation[cls.id] = dict(
                        name=cls.display_name, **stats)
            except Exception:
                continue
            if fb.isCanceled():
                return
        if result.cross_validation and result.total:
            best = min(result.cross_validation.items(),
                       key=lambda kv: kv[1]["rmse"])
            used = self.engine_id or interp.DEFAULT_ENGINE.get(kind)
            if used in result.cross_validation:
                used_rmse = result.cross_validation[used]["rmse"]
                if best[1]["rmse"] < used_rmse * 0.75:
                    result.add_issue(
                        QualityIssue.SEVERITY_INFO,
                        "The %s engine cross-validates markedly better "
                        "(RMSE %.3f) than the engine used (RMSE %.3f). "
                        "Consider re-running with it."
                        % (best[1]["name"], best[1]["rmse"], used_rmse))
                rmse = result.cross_validation[used]["rmse"]
                result.volume_uncertainty = vol_mod.volume_uncertainty(
                    rmse, result.total.area, self.source.x.size)

    def _compare_tin(self, result):
        if not result.tin_check or not result.total:
            return
        for name in ("cut", "fill"):
            grid_v = getattr(result.total, name)
            tin_v = getattr(result.tin_check, name)
            if grid_v <= 0:
                continue
            diff = abs(grid_v - tin_v) / grid_v * 100.0
            if diff > 2.0:
                result.add_issue(
                    QualityIssue.SEVERITY_WARNING,
                    "The grid and exact TIN integrations differ by %.1f%% on "
                    "the %s volume (%.0f against %.0f). This normally means "
                    "the cell size is too coarse for the terrain. Refine the "
                    "grid before issuing the quantities."
                    % (diff, name, grid_v, tin_v))

    def _convergence(self, result, engine, fb):
        base_cell = result.grid.cell

        def compute(cell):
            g = build_grid(self.source, self.aoi_layer, cell)
            _, _, w = coverage_weights(g, self.aoi_layer,
                                       subsample=max(2, self.subsample // 2),
                                       selected_only=self.selected_only)
            eng = engine.clone() if engine is not None else None
            ground, _ = grid_from_source(self.source, g, eng)
            strip = self.soil.topsoil_strip if self.soil else 0.0
            d = vol_mod.depth_array(ground - strip, self.design.to_grid(g))
            return vol_mod.integrate(d, w, g.cell_area)

        n_cells = result.grid.ncols * result.grid.nrows
        factors = (1, 2) if n_cells > 2_000_000 else (1, 2, 4)
        try:
            rows, verdict = vol_mod.convergence_test(compute, base_cell, factors)
        except Exception:
            return
        result.convergence = (rows, verdict)
        if verdict and not verdict.get("converged"):
            result.add_issue(
                QualityIssue.SEVERITY_WARNING,
                "Quantities are still moving as the grid is refined (cut "
                "changes by %.1f%%, fill by %.1f%% between the coarsest and "
                "finest test grids). Use a finer cell size for the reported "
                "figures." % (verdict["cut"], verdict["fill"]))

    # ------------------------------------------------------------------
    def _write_outputs(self, result, fb):
        if not self.output_dir:
            return
        grid = result.grid
        written = result.written

        if self.write_rasters:
            p = self._path("cutfill_depth", "tif")
            out_mod.write_depth_raster(p, result.depth, result.weight, grid)
            written["depth_raster"] = p
            if self.add_to_project:
                out_mod.load_raster(p, "%s - cut/fill depth" % self.prefix,
                                    style_cut_fill=True)
            pg = self._path("existing_surface", "tif")
            out_mod.write_surface_raster(pg, result.ground, grid)
            written["ground_raster"] = pg
            pd = self._path("design_surface", "tif")
            out_mod.write_surface_raster(pd, result.design_grid, grid)
            written["design_raster"] = pd

        if self.write_zones:
            try:
                p = self._path("zones", "gpkg")
                out_mod.write_zone_polygons(p, result.depth, result.weight, grid,
                                            tolerance=self.zone_tolerance,
                                            add_to_project=self.add_to_project)
                written["zones"] = p
            except Exception as exc:
                result.add_issue(QualityIssue.SEVERITY_WARNING,
                                 "Zone polygons could not be created: %s" % exc)

        if self.write_contours and written.get("depth_raster"):
            try:
                interval = self.contour_interval or max(
                    round((result.total.max_cut + result.total.max_fill) / 10.0, 2),
                    0.1)
                p = self._path("depth_contours", "gpkg")
                out_mod.write_depth_contours(p, written["depth_raster"], interval,
                                             add_to_project=self.add_to_project)
                written["contours"] = p
            except Exception as exc:
                result.add_issue(QualityIssue.SEVERITY_WARNING,
                                 "Depth contours could not be created: %s" % exc)

        if self.write_csv:
            p = self._path("quantities", "csv")
            out_mod.write_results_csv(p, result.per_feature, result.mapping,
                                      self.soil, unit=self.unit)
            written["quantities_csv"] = p
            if result.curve is not None:
                pc = self._path("balance_curve", "csv")
                out_mod.write_curve_csv(pc, result.curve)
                written["curve_csv"] = pc

        if self.write_chart and result.curve is not None:
            p = self._path("balance_curve", "png")
            got = out_mod.write_balance_chart(
                p, result.curve, balance_level=result.balance_level,
                chosen_level=getattr(result, "chosen_level", None),
                unit=self.unit)
            if got:
                written["balance_chart"] = got
            else:
                result.add_issue(
                    QualityIssue.SEVERITY_INFO,
                    "Matplotlib is not available, so the balance chart was not "
                    "rendered. The curve data was still written to CSV.")
            ph = self._path("hypsometry", "png")
            got_h = out_mod.write_hypsometry_chart(
                ph, result.ground_stripped, result.weight, grid.cell_area,
                unit=self.unit)
            if got_h:
                written["hypsometry_chart"] = got_h

        if self.write_back_attributes:
            ok, msg = out_mod.write_results_to_layer(
                self.aoi_layer, result.per_feature, result.mapping, self.soil)
            if not ok and msg:
                result.add_issue(QualityIssue.SEVERITY_INFO, msg)

    # ------------------------------------------------------------------
    def _settings_summary(self, engine, cell, recommended):
        rows = list(self.source.describe())
        rows += [("Area of interest", self.aoi_layer.name()),
                 ("Polygons used",
                  "selected features only" if self.selected_only else "all features")]
        rows += self.design.describe()
        rows += [("Interpolation engine",
                  engine.display_name if engine else "Direct DEM resampling")]
        if engine is not None and engine.params:
            rows.append(("Engine parameters",
                         ", ".join("%s = %s" % kv for kv in sorted(engine.params.items()))))
        rows += [("Recommended cell size", "%.3f" % recommended)]
        rows += self.grid_rows(cell)
        rows += [("Boundary subsampling", "%d x %d per cell"
                  % (self.subsample, self.subsample))]
        if self.soil is not None:
            rows += self.soil.summary()
        return rows

    def grid_rows(self, cell):
        return [("Cell size used", "%.3f" % cell)]
