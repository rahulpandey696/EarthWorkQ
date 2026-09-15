# -*- coding: utf-8 -*-
"""
EarthWorkQ - earthwork cut and fill quantities for QGIS.

Plugin entry point.  The GUI collects settings, hands them to
core.analysis.EarthworkAnalysis, and runs it inside a QgsTask so that QGIS
stays responsive and the run can be cancelled.
"""

from __future__ import annotations

import os
import traceback

from qgis.core import QgsApplication, QgsMessageLog, QgsProject, QgsTask
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QTranslator, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon

# QAction moved from QtWidgets to QtGui in Qt6. The qgis.PyQt shim usually
# papers over this, but not on every build.
try:
    from qgis.PyQt.QtWidgets import QAction
except ImportError:  # pragma: no cover
    from qgis.PyQt.QtGui import QAction

from .core import compat
from .core import interpolation as interp
from .core import report as report_mod
from .core.analysis import AnalysisError, EarthworkAnalysis
from .core.design import DesignRaster, FlatLevel, SlopedPlane
from .core.sources import (
    RasterSource,
    SourceError,
    load_contour_layer,
    load_point_layer,
    load_table,
)
from .gui.dialog import (
    DESIGN_DEM,
    DESIGN_FLAT,
    DESIGN_PLANE,
    SOURCE_CONTOURS,
    SOURCE_DEM,
    SOURCE_POINTS,
    SOURCE_TABLE,
    EarthWorkQDialog,
)

MESSAGE_CATEGORY = "EarthWorkQ"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


class TaskFeedback(object):
    """Adapter so the analysis can report into a QgsTask."""

    def __init__(self, task):
        self.task = task

    def setProgress(self, value):
        self.task.setProgress(float(value))

    def pushInfo(self, text):
        self.task.log(text)

    def isCanceled(self):
        return self.task.isCanceled()


class EarthworkTask(QgsTask):

    def __init__(self, analysis, description="EarthWorkQ calculation"):
        flags = compat.TASK_CAN_CANCEL
        if flags is None:
            super(EarthworkTask, self).__init__(description)
        else:
            super(EarthworkTask, self).__init__(description, flags)
        self.analysis = analysis
        self.result = None
        self.exception = None
        self.messages = []

    def log(self, text):
        self.messages.append(str(text))
        QgsMessageLog.logMessage(str(text), MESSAGE_CATEGORY, compat.LEVEL_INFO)

    def run(self):
        try:
            self.result = self.analysis.run(TaskFeedback(self))
            return not self.isCanceled()
        except Exception as exc:                     # noqa: BLE001
            self.exception = exc
            self.log(traceback.format_exc())
            return False


class EarthWorkQPlugin(object):
    """QGIS plugin implementation."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = PLUGIN_DIR
        self.actions = []
        self.menu = "&EarthWorkQ"
        self.dlg = None
        self.task = None
        self.provider = None
        self._last_report = None

        locale = (QSettings().value("locale/userLocale") or "en")[0:2]
        qm = os.path.join(self.plugin_dir, "i18n", "EarthWorkQ_%s.qm" % locale)
        if os.path.exists(qm):
            self.translator = QTranslator()
            self.translator.load(qm)
            QCoreApplication.installTranslator(self.translator)

    # ------------------------------------------------------------------
    @staticmethod
    def tr(message):
        return QCoreApplication.translate("EarthWorkQ", message)

    def initGui(self):
        icon = QIcon(os.path.join(self.plugin_dir, "icon.png"))
        action = QAction(icon, self.tr("Earthwork quantities (EarthWorkQ)"),
                         self.iface.mainWindow())
        action.triggered.connect(self.run)
        self.iface.addToolBarIcon(action)
        try:
            self.iface.addPluginToRasterMenu(self.menu, action)
        except AttributeError:
            self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)
        self._init_processing()

    def _init_processing(self):
        try:
            from .processing_alg.provider import EarthWorkQProvider
            self.provider = EarthWorkQProvider()
            QgsApplication.processingRegistry().addProvider(self.provider)
        except Exception:
            QgsMessageLog.logMessage(
                "The EarthWorkQ Processing provider could not be registered:\n"
                + traceback.format_exc(), MESSAGE_CATEGORY, compat.LEVEL_WARNING)

    def unload(self):
        for action in self.actions:
            try:
                self.iface.removePluginRasterMenu(self.menu, action)
            except AttributeError:
                self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        self.actions = []
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None

    # ------------------------------------------------------------------
    def run(self):
        self.dlg = EarthWorkQDialog(self.iface.mainWindow())
        self.dlg.runRequested.connect(self._start)
        self.dlg.cancelRequested.connect(self._cancel)
        self.dlg.btn_open_report.clicked.connect(self._open_report)
        self.dlg.show()

    # ------------------------------------------------------------------
    def _build_source(self):
        d = self.dlg
        kind = d.cmb_source_type.currentIndex()

        if kind == SOURCE_TABLE:
            path = d.txt_table_path.text().strip()
            if not path or not os.path.exists(path):
                raise SourceError("Select an existing CSV or Excel file of "
                                  "coordinates.")
            crs = d.crs_table.crs() if d.crs_table is not None else None
            return load_table(path,
                              easting_field=d.cmb_field_east.currentText() or None,
                              northing_field=d.cmb_field_north.currentText() or None,
                              elevation_field=d.cmb_field_elev.currentText() or None,
                              crs=crs)

        if kind == SOURCE_POINTS:
            layer = d.cmb_point_layer.currentLayer()
            if layer is None:
                raise SourceError("Select a point layer.")
            return load_point_layer(layer, d.cmb_point_field.currentText(),
                                    use_z=d.chk_use_geometry_z.isChecked())

        if kind == SOURCE_CONTOURS:
            layer = d.cmb_contour_layer.currentLayer()
            if layer is None:
                raise SourceError("Select a contour layer.")
            spacing = d.spn_densify.value() or None
            return load_contour_layer(
                layer, d.cmb_contour_field.currentText(),
                densify_spacing=spacing,
                check_topology=d.chk_contour_topology.isChecked())

        layer = d.cmb_dem_layer.currentLayer()
        if layer is None:
            raise SourceError("Select a DEM layer.")
        return RasterSource(layer, d.spn_dem_band.value())

    def _build_design(self, source, aoi):
        d = self.dlg
        kind = d.cmb_design_type.currentIndex()
        if kind == DESIGN_FLAT:
            return FlatLevel(d.spn_level.value())
        if kind == DESIGN_PLANE:
            x0 = d.spn_plane_x.value()
            y0 = d.spn_plane_y.value()
            if x0 == 0.0 and y0 == 0.0:
                c = aoi.extent().center()
                x0, y0 = c.x(), c.y()
            return SlopedPlane(d.spn_plane_z.value(), x0, y0,
                               d.spn_grade_x.value() / 100.0,
                               d.spn_grade_y.value() / 100.0)
        layer = d.cmb_design_dem.currentLayer()
        if layer is None:
            raise SourceError("Select a design DEM.")
        return DesignRaster(RasterSource(layer, 1), d.spn_design_offset.value())

    def _start(self):
        d = self.dlg
        d.log.clear()
        try:
            source = self._build_source()
            aoi = d.cmb_aoi.currentLayer()
            if aoi is None:
                raise SourceError("Select a polygon layer for the area of "
                                  "interest.")
            if d.chk_selected_only.isChecked() and aoi.selectedFeatureCount() == 0:
                raise SourceError("'Use selected features only' is ticked but "
                                  "nothing is selected in '%s'." % aoi.name())
            design = self._build_design(source, aoi)
            soil = d.build_soil()
        except Exception as exc:                     # noqa: BLE001
            d.error(str(exc))
            return

        if hasattr(source, "flag_outliers"):
            try:
                source.flag_outliers()
            except Exception:
                pass

        out_dir = d.txt_output_dir.text().strip() or None
        if out_dir is None:
            d.warn("No output folder was set, so only the on-screen summary "
                   "will be produced. Set a folder on the Outputs tab to keep "
                   "the rasters, tables and report.")

        # solve for the balance level first if asked
        analysis = EarthworkAnalysis(
            source=source,
            aoi_layer=aoi,
            design=design,
            soil=soil,
            engine_id=d.cmb_engine.currentData(),
            engine_params=d.engine_params(),
            cell_size=(d.spn_cell.value() or None),
            subsample=d.spn_subsample.value(),
            selected_only=d.chk_selected_only.isChecked(),
            run_cross_validation=d.chk_cross_validate.isChecked(),
            run_convergence=d.chk_convergence.isChecked(),
            balance_steps=d.spn_balance_steps.value(),
            zone_tolerance=d.spn_zone_tol.value(),
            contour_interval=(d.spn_contour_interval.value() or None),
            output_dir=out_dir,
            prefix=d.txt_prefix.text().strip() or "earthworkq",
            write_rasters=d.chk_out_raster.isChecked(),
            write_zones=d.chk_out_zones.isChecked(),
            write_contours=d.chk_out_contours.isChecked(),
            write_chart=d.chk_out_chart.isChecked(),
            write_csv=d.chk_out_csv.isChecked(),
            write_back_attributes=d.chk_write_back.isChecked(),
            add_to_project=d.chk_add_layers.isChecked(),
            solve_balance_first=(
                d.cmb_design_type.currentIndex() == DESIGN_FLAT
                and d.chk_use_balance_level.isChecked()),
        )

        d.set_running(True)
        d.progress.setValue(0)
        d.append_log("EarthWorkQ starting.")
        d.append_log("Engine: %s" % (d.cmb_engine.currentText()))

        self.task = EarthworkTask(analysis)
        self.task.progressChanged.connect(
            lambda p: d.progress.setValue(int(p)))
        self.task.taskCompleted.connect(self._finished)
        self.task.taskTerminated.connect(self._finished)
        QgsApplication.taskManager().addTask(self.task)

    def _cancel(self):
        if self.task is not None:
            self.task.cancel()
        self.dlg.append_log("Cancellation requested.")

    # ------------------------------------------------------------------
    def _finished(self):
        d = self.dlg
        task = self.task
        d.set_running(False)
        for m in getattr(task, "messages", []):
            d.append_log("  " + m)

        if task is None:
            return
        if task.exception is not None:
            d.progress.setValue(0)
            d.append_log("FAILED: %s" % task.exception)
            d.error("The calculation failed:\n\n%s" % task.exception)
            self.task = None
            return
        if task.result is None:
            d.append_log("Cancelled - no output was written.")
            d.progress.setValue(0)
            self.task = None
            return

        result = task.result
        self._summarise(result, task.analysis)
        self._write_report(result, task.analysis)
        d.progress.setValue(100)
        self.task = None

    def _summarise(self, result, analysis):
        d = self.dlg
        t = result.total
        u = result.unit
        soil = analysis.soil
        d.append_log("")
        d.append_log("=" * 58)
        d.append_log("RESULTS")
        d.append_log("=" * 58)
        d.append_log("Cut (in situ)          : %15.2f %s3" % (t.cut, u))
        d.append_log("Fill (compacted)       : %15.2f %s3" % (t.fill, u))
        if soil is not None:
            need = soil.cut_for_fill(t.fill)
            d.append_log("Cut needed for fill    : %15.2f %s3  "
                         "(shrinkage factor %.4f)" % (need, u, soil.shrinkage_factor))
            surplus = t.cut - need
            d.append_log("%-22s : %15.2f %s3"
                         % ("Surplus" if surplus >= 0 else "Deficit",
                            abs(surplus), u))
        else:
            d.append_log("Net (cut - fill)       : %15.2f %s3" % (t.net, u))
        d.append_log("Area computed          : %15.2f %s2" % (t.area, u))
        if t.nodata_area > 0:
            d.append_log("Area with no elevation : %15.2f %s2" % (t.nodata_area, u))
        d.append_log("Max cut / max fill     : %8.3f / %8.3f %s"
                     % (t.max_cut, t.max_fill, u))
        if result.topsoil_volume:
            d.append_log("Topsoil stripped       : %15.2f %s3"
                         % (result.topsoil_volume, u))
        if result.balance_level is not None:
            d.append_log("Balancing level        : %15.3f %s" % (result.balance_level, u))
        if result.minimum_movement_level is not None:
            d.append_log("Minimum movement level : %15.3f %s"
                         % (result.minimum_movement_level, u))
        if result.volume_uncertainty:
            d.append_log("Volume uncertainty     : +/- %11.2f %s3 (interpolation only)"
                         % (result.volume_uncertainty, u))

        if result.cross_validation:
            d.append_log("")
            d.append_log("Interpolation cross-validation (RMSE, %s):" % u)
            for eid, s in sorted(result.cross_validation.items(),
                                 key=lambda kv: kv[1]["rmse"]):
                mark = "  <- used" if eid == (analysis.engine_id or "") else ""
                d.append_log("  %-46s %8.4f%s" % (s["name"], s["rmse"], mark))

        if result.tin_check is not None:
            d.append_log("")
            d.append_log("Independent TIN integration: cut %.2f, fill %.2f"
                         % (result.tin_check.cut, result.tin_check.fill))

        if result.issues:
            d.append_log("")
            d.append_log("Notes:")
            for issue in result.issues:
                d.append_log("  %s" % issue)

        if result.written:
            d.append_log("")
            d.append_log("Files written:")
            for k, v in sorted(result.written.items()):
                d.append_log("  %-18s %s" % (k, v))

    def _write_report(self, result, analysis):
        d = self.dlg
        if not d.chk_out_report.isChecked() or not analysis.output_dir:
            return
        path = os.path.join(analysis.output_dir,
                            "%s_report.html" % analysis.prefix)
        try:
            report_mod.write_report(
                path, result, analysis,
                project_name=d.txt_project.text().strip() or None,
                prepared_by=d.txt_prepared_by.text().strip() or None,
                notes=d.txt_notes.toPlainText().strip() or None)
        except Exception as exc:                     # noqa: BLE001
            d.append_log("The report could not be written: %s" % exc)
            return
        result.written["report"] = path
        self._last_report = path
        d.btn_open_report.setEnabled(True)
        d.append_log("  %-18s %s" % ("report", path))

    def _open_report(self):
        if self._last_report and os.path.exists(self._last_report):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_report))
