# -*- coding: utf-8 -*-
"""EarthWorkQ Processing algorithms."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
)

from ..core import balance as balance_mod
from ..core import compat
from ..core import interpolation as interp
from ..core import outputs as out_mod
from ..core import report as report_mod
from ..core.analysis import EarthworkAnalysis
from ..core.design import DesignRaster, FlatLevel
from ..core.grid import build_grid, coverage_weights, grid_from_source, write_raster
from ..core.soil import SoilModel
from ..core.sources import (
    RasterSource,
    load_contour_layer,
    load_point_layer,
)

ENGINE_CHOICES = [
    ("Automatic (recommended for the input type)", None),
    ("TIN - cubic (Clough-Tocher)", "tin_cubic"),
    ("TIN - linear (Delaunay)", "tin_linear"),
    ("Spline - thin plate with tension", "spline_rbf"),
    ("Inverse distance weighting", "idw"),
    ("GRASS r.surf.contour (contours only)", "grass_contour"),
    ("GRASS v.surf.rst", "grass_rst"),
    ("QGIS TIN interpolation", "qgis_tin"),
]

SOURCE_CHOICES = ["Point layer", "Contour layer", "DEM raster"]


class _Base(QgsProcessingAlgorithm):

    def createInstance(self):
        return self.__class__()

    def group(self):
        return "Earthworks"

    def groupId(self):
        return "earthworks"

    def _build_source(self, parameters, context, feedback):
        idx = self.parameterAsEnum(parameters, "SOURCE_TYPE", context)
        if idx == 2:
            layer = self.parameterAsRasterLayer(parameters, "DEM", context)
            if layer is None:
                raise QgsProcessingException(
                    "A DEM layer is required when the source type is 'DEM raster'.")
            return RasterSource(layer, 1)

        layer = self.parameterAsVectorLayer(parameters, "SURVEY", context)
        if layer is None:
            raise QgsProcessingException(
                "A point or contour layer is required for this source type.")
        field = self.parameterAsString(parameters, "ELEVATION_FIELD", context)
        if idx == 0:
            return load_point_layer(layer, field)
        return load_contour_layer(layer, field)

    def _build_soil(self, parameters, context):
        if not self.parameterAsBool(parameters, "APPLY_MATERIAL", context):
            return None
        return SoilModel(
            e_insitu=self.parameterAsDouble(parameters, "E_INSITU", context),
            e_compacted=self.parameterAsDouble(parameters, "E_COMPACTED", context),
            gs=self.parameterAsDouble(parameters, "GS", context),
            topsoil_strip=self.parameterAsDouble(parameters, "TOPSOIL", context),
            name="Processing input")

    def _common_inputs(self, with_material=True):
        self.addParameter(QgsProcessingParameterEnum(
            "SOURCE_TYPE", "Existing ground from", options=SOURCE_CHOICES,
            defaultValue=0))
        self.addParameter(QgsProcessingParameterVectorLayer(
            "SURVEY", "Survey layer (points or contours)",
            types=[t for t in (compat.processing_source_type("VectorPoint"),
                               compat.processing_source_type("VectorLine"))
                   if t is not None],
            optional=True))
        self.addParameter(QgsProcessingParameterField(
            "ELEVATION_FIELD", "Elevation attribute",
            parentLayerParameterName="SURVEY",
            type=compat.processing_field_type("Numeric"), optional=True))
        self.addParameter(QgsProcessingParameterRasterLayer(
            "DEM", "Existing DEM", optional=True))
        self.addParameter(QgsProcessingParameterVectorLayer(
            "AREA", "Area of interest",
            types=[t for t in (compat.processing_source_type("VectorPolygon"),)
                   if t is not None]))
        self.addParameter(QgsProcessingParameterEnum(
            "ENGINE", "Interpolation engine",
            options=[c[0] for c in ENGINE_CHOICES], defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            "CELL_SIZE", "Grid cell size (0 = recommended from data density)",
            type=compat.processing_number_type("Double"), defaultValue=0.0,
            minValue=0.0))
        self.addParameter(QgsProcessingParameterNumber(
            "SUBSAMPLE", "Boundary subsampling factor",
            type=compat.processing_number_type("Integer"), defaultValue=8,
            minValue=1, maxValue=32))
        if with_material:
            self.addParameter(QgsProcessingParameterBoolean(
                "APPLY_MATERIAL", "Apply material shrinkage", defaultValue=True))
            self.addParameter(QgsProcessingParameterNumber(
                "E_INSITU", "In-situ void ratio",
                type=compat.processing_number_type("Double"), defaultValue=0.70,
                minValue=0.01, maxValue=3.0))
            self.addParameter(QgsProcessingParameterNumber(
                "E_COMPACTED", "Compacted void ratio",
                type=compat.processing_number_type("Double"), defaultValue=0.55,
                minValue=0.01, maxValue=3.0))
            self.addParameter(QgsProcessingParameterNumber(
                "GS", "Particle specific gravity",
                type=compat.processing_number_type("Double"), defaultValue=2.65,
                minValue=2.0, maxValue=3.5))
            self.addParameter(QgsProcessingParameterNumber(
                "TOPSOIL", "Topsoil strip depth",
                type=compat.processing_number_type("Double"), defaultValue=0.0,
                minValue=0.0))

    def _engine_id(self, parameters, context):
        return ENGINE_CHOICES[self.parameterAsEnum(parameters, "ENGINE", context)][1]


# ----------------------------------------------------------------------
class EarthworkQuantitiesAlgorithm(_Base):

    def name(self):
        return "earthworkquantities"

    def displayName(self):
        return "Earthwork quantities (cut and fill)"

    def shortHelpString(self):
        return (
            "Computes cut and fill quantities between a surveyed existing "
            "ground surface and a design formation, over one or more "
            "polygons.\n\n"
            "Boundary cells are area-weighted rather than counted whole. "
            "Volumes are converted between cut and fill using the shrinkage "
            "factor derived from the in-situ and compacted void ratios, so "
            "the surplus reported is the volume genuinely left over once "
            "enough material has been dug to build the fill.\n\n"
            "An HTML report recording every input, assumption and quality "
            "check is written alongside the numerical outputs.")

    def initAlgorithm(self, config=None):
        self._common_inputs()
        self.addParameter(QgsProcessingParameterNumber(
            "FORMATION_LEVEL", "Formation level",
            type=compat.processing_number_type("Double"), defaultValue=0.0))
        self.addParameter(QgsProcessingParameterBoolean(
            "SOLVE_BALANCE", "Solve for the balancing level instead",
            defaultValue=False))
        self.addParameter(QgsProcessingParameterRasterLayer(
            "DESIGN_DEM", "Design DEM (overrides the formation level)",
            optional=True))
        self.addParameter(QgsProcessingParameterBoolean(
            "VERIFY", "Run cross-validation and convergence checks",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterFolderDestination(
            "OUTPUT_FOLDER", "Output folder"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self._build_source(parameters, context, feedback)
        aoi = self.parameterAsVectorLayer(parameters, "AREA", context)
        soil = self._build_soil(parameters, context)

        design_dem = self.parameterAsRasterLayer(parameters, "DESIGN_DEM", context)
        if design_dem is not None:
            design = DesignRaster(RasterSource(design_dem, 1))
        else:
            design = FlatLevel(
                self.parameterAsDouble(parameters, "FORMATION_LEVEL", context))

        folder = self.parameterAsString(parameters, "OUTPUT_FOLDER", context)
        verify = self.parameterAsBool(parameters, "VERIFY", context)

        analysis = EarthworkAnalysis(
            source=source, aoi_layer=aoi, design=design, soil=soil,
            engine_id=self._engine_id(parameters, context),
            cell_size=(self.parameterAsDouble(parameters, "CELL_SIZE", context) or None),
            subsample=self.parameterAsInt(parameters, "SUBSAMPLE", context),
            run_cross_validation=verify, run_convergence=verify,
            output_dir=folder, add_to_project=False,
            solve_balance_first=self.parameterAsBool(
                parameters, "SOLVE_BALANCE", context))

        result = analysis.run(feedback)

        report_path = os.path.join(folder, "%s_report.html" % analysis.prefix)
        try:
            report_mod.write_report(report_path, result, analysis)
            result.written["report"] = report_path
        except Exception as exc:                     # noqa: BLE001
            feedback.reportError("The report could not be written: %s" % exc)

        t = result.total
        feedback.pushInfo("")
        feedback.pushInfo("Cut  : %.2f" % t.cut)
        feedback.pushInfo("Fill : %.2f" % t.fill)
        if soil is not None:
            feedback.pushInfo("Surplus after shrinkage : %.2f"
                              % t.net_after_shrinkage(soil))
        if result.balance_level is not None:
            feedback.pushInfo("Balancing level : %.3f" % result.balance_level)
        for issue in result.issues:
            (feedback.reportError if issue.severity == "error"
             else feedback.pushWarning if hasattr(feedback, "pushWarning")
             else feedback.pushInfo)(str(issue))

        return {
            "OUTPUT_FOLDER": folder,
            "CUT": t.cut,
            "FILL": t.fill,
            "NET": t.net_after_shrinkage(soil),
            "AREA": t.area,
            "BALANCE_LEVEL": result.balance_level,
            "REPORT": result.written.get("report"),
        }


# ----------------------------------------------------------------------
class BalanceLevelAlgorithm(_Base):

    def name(self):
        return "balancelevel"

    def displayName(self):
        return "Balance level analysis"

    def shortHelpString(self):
        return (
            "Sweeps the formation level across the full range of ground "
            "elevations and reports cut, fill and the net after shrinkage at "
            "each, together with the level at which the site balances and the "
            "level of minimum total earthmoving.\n\n"
            "The two are rarely the same, and choosing between them is a cost "
            "decision: hauling material off site against double-handling it "
            "on site.")

    def initAlgorithm(self, config=None):
        self._common_inputs()
        self.addParameter(QgsProcessingParameterNumber(
            "STEPS", "Number of levels to test",
            type=compat.processing_number_type("Integer"), defaultValue=200,
            minValue=20, maxValue=5000))
        self.addParameter(QgsProcessingParameterFileDestination(
            "OUTPUT_CSV", "Balance curve (CSV)", fileFilter="CSV files (*.csv)"))
        self.addParameter(QgsProcessingParameterFileDestination(
            "OUTPUT_CHART", "Balance chart (PNG)",
            fileFilter="PNG files (*.png)", optional=True))

    def processAlgorithm(self, parameters, context, feedback):
        source = self._build_source(parameters, context, feedback)
        aoi = self.parameterAsVectorLayer(parameters, "AREA", context)
        soil = self._build_soil(parameters, context)
        sf = soil.shrinkage_factor if soil else 1.0

        cell = self.parameterAsDouble(parameters, "CELL_SIZE", context) or None
        grid = build_grid(source, aoi, cell)
        feedback.pushInfo("Grid: %d x %d at %.3f" % (grid.ncols, grid.nrows, grid.cell))

        _, _, weight = coverage_weights(
            grid, aoi, subsample=self.parameterAsInt(parameters, "SUBSAMPLE", context))

        engine_id = self._engine_id(parameters, context)
        engine = None
        if not isinstance(source, RasterSource):
            kind = getattr(source, "kind", "points")
            engine = interp.create_engine(
                engine_id or interp.DEFAULT_ENGINE.get(kind, "tin_linear"))
        ground, _ = grid_from_source(source, grid, engine, feedback=feedback)
        if soil and soil.topsoil_strip:
            ground = ground - soil.topsoil_strip

        curve, _ = balance_mod.sweep_levels(
            ground, weight, grid.cell_area,
            n_steps=self.parameterAsInt(parameters, "STEPS", context),
            shrinkage_factor=sf)
        if curve is None:
            raise QgsProcessingException(
                "No ground elevations fall inside the area of interest.")

        csv_path = self.parameterAsFileOutput(parameters, "OUTPUT_CSV", context)
        out_mod.write_curve_csv(csv_path, curve)

        chart_path = self.parameterAsFileOutput(parameters, "OUTPUT_CHART", context)
        if chart_path:
            got = out_mod.write_balance_chart(
                chart_path, curve, balance_level=curve.balance_level())
            if not got:
                feedback.pushInfo("Matplotlib is unavailable, so no chart was "
                                  "rendered.")

        level = curve.balance_level()
        feedback.pushInfo("Shrinkage factor applied : %.4f" % sf)
        feedback.pushInfo("Balancing level          : %s"
                          % ("%.3f" % level if level is not None
                             else "none within the ground range"))
        feedback.pushInfo("Minimum movement level   : %.3f"
                          % curve.minimum_movement_level())

        return {"OUTPUT_CSV": csv_path, "OUTPUT_CHART": chart_path,
                "BALANCE_LEVEL": level,
                "MIN_MOVEMENT_LEVEL": curve.minimum_movement_level()}


# ----------------------------------------------------------------------
class SurfaceFromSurveyAlgorithm(_Base):

    def name(self):
        return "surfacefromsurvey"

    def displayName(self):
        return "Build a surface from survey data"

    def shortHelpString(self):
        return (
            "Interpolates a DEM from survey points or contour lines without "
            "extrapolating beyond the surveyed area, and reports the "
            "cross-validated error of every applicable engine so the choice of "
            "interpolator can be justified rather than assumed.")

    def initAlgorithm(self, config=None):
        self._common_inputs(with_material=False)
        self.addParameter(QgsProcessingParameterBoolean(
            "CROSS_VALIDATE", "Cross-validate every applicable engine",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterRasterDestination(
            "OUTPUT", "Interpolated surface"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self._build_source(parameters, context, feedback)
        aoi = self.parameterAsVectorLayer(parameters, "AREA", context)
        cell = self.parameterAsDouble(parameters, "CELL_SIZE", context) or None
        grid = build_grid(source, aoi, cell)

        engine = None
        if not isinstance(source, RasterSource):
            kind = getattr(source, "kind", "points")
            engine = interp.create_engine(
                self._engine_id(parameters, context)
                or interp.DEFAULT_ENGINE.get(kind, "tin_linear"))
            feedback.pushInfo("Engine: %s" % engine.display_name)

        if self.parameterAsBool(parameters, "CROSS_VALIDATE", context) \
                and not isinstance(source, RasterSource) \
                and interp.scipy_available():
            feedback.pushInfo("")
            feedback.pushInfo("Cross-validated RMSE by engine:")
            kind = getattr(source, "kind", "points")
            for cls in interp.engines_for(kind):
                if getattr(cls, "produces_raster", False):
                    continue
                try:
                    stats = cls().cross_validate(source.x, source.y, source.z)
                except Exception:
                    continue
                if stats:
                    feedback.pushInfo("  %-46s %8.4f"
                                      % (cls.display_name, stats["rmse"]))

        ground, _ = grid_from_source(source, grid, engine, feedback=feedback)
        out = self.parameterAsOutputLayer(parameters, "OUTPUT", context)
        write_raster(out, ground, grid)
        return {"OUTPUT": out}
