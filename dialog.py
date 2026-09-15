# -*- coding: utf-8 -*-
"""
EarthWorkQ main dialog.

Built in code rather than from a .ui file so the widget set can adapt to what
is actually available in the running QGIS - SciPy, GRASS, matplotlib, openpyxl
- instead of offering options that will fail at run time.

Layout rules that matter on a scaled or small display:

  * every tab's contents sit inside a QScrollArea, so the dialog can be made
    smaller than its natural content height without anything being pushed off
    the bottom of the screen;
  * the opening size is derived from the screen's available geometry - which
    already excludes the task bar - rather than being hard coded, so the window
    fits the work area even at 150% display scaling;
  * the progress bar and button row live outside the scroll area and are
    therefore always reachable.
"""

from __future__ import annotations

import os

from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices, QGuiApplication, QIcon
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import compat
from ..core import interpolation as interp
from ..core.soil import MATERIAL_PRESETS

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_TABLE = 0
SOURCE_POINTS = 1
SOURCE_CONTOURS = 2
SOURCE_DEM = 3

DESIGN_FLAT = 0
DESIGN_PLANE = 1
DESIGN_DEM = 2

VECTOR_FILTER = ("Vector layers (*.shp *.gpkg *.geojson *.json *.kml *.gml "
                 "*.tab *.dxf);;All files (*)")
RASTER_FILTER = ("Raster layers (*.tif *.tiff *.asc *.img *.vrt *.dem *.bil "
                 "*.hgt *.grd);;All files (*)")


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _qt_flag(name):
    """Resolve a Qt window flag across Qt5 (unscoped) and Qt6 (scoped)."""
    scope = getattr(Qt, "WindowType", Qt)
    return getattr(scope, name, None)


def _hline():
    f = QFrame()
    try:
        f.setFrameShape(QFrame.Shape.HLine)
        f.setFrameShadow(QFrame.Shadow.Sunken)
    except AttributeError:
        f.setFrameShape(QFrame.HLine)
        f.setFrameShadow(QFrame.Sunken)
    return f


def _size_policy(name):
    """QSizePolicy enumerators are scoped in Qt6 and unscoped in Qt5."""
    scope = getattr(QSizePolicy, "Policy", QSizePolicy)
    return getattr(scope, name)


def _no_frame(widget):
    try:
        widget.setFrameShape(QFrame.Shape.NoFrame)
    except AttributeError:
        widget.setFrameShape(QFrame.NoFrame)
    return widget


def _hint(text):
    """Small grey explanatory paragraph placed under a group of controls."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: palette(mid); font-size: 11px;")
    return lbl


def _banner(text):
    """The description at the top of each tab, explaining what the tab is for."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        "background: palette(alternate-base);"
        "border: 1px solid palette(mid);"
        "border-radius: 6px; padding: 9px 11px; font-size: 11px;")
    return lbl


def _scrollable(widget):
    """Wrap a tab page so it scrolls instead of forcing the dialog taller."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    _no_frame(area)
    area.setWidget(widget)
    return area


def _tip(widget, text):
    """Set a tooltip, including on the internal editor of editable widgets.

    An editable combo box or a spin box contains its own QLineEdit, and that
    child is what sits under the cursor when the user hovers the text area.
    """
    widget.setToolTip(text)
    editor = getattr(widget, "lineEdit", None)
    if callable(editor):
        try:
            child = editor()
        except TypeError:
            child = None
        if child is not None:
            child.setToolTip(text)


def _add(form, label_text, widget, tip=None):
    """Add a form row, applying the same tooltip to the label and the field."""
    label = QLabel(label_text)
    if tip:
        label.setToolTip(tip)
        if isinstance(widget, QWidget):
            _tip(widget, tip)
        else:                       # it's a layout of several widgets
            for i in range(widget.count()):
                item = widget.itemAt(i).widget()
                if item is not None and not item.toolTip():
                    _tip(item, tip)
    form.addRow(label, widget)
    return widget


def _check(text, tip=None, checked=False):
    box = QCheckBox(text)
    box.setChecked(checked)
    if tip:
        box.setToolTip(tip)
    return box


# ----------------------------------------------------------------------
class LayerInput(QWidget):
    """A layer chooser with a Browse button for loading one straight from disk.

    Exposes the parts of the wrapped combo box that calling code uses, so it is
    a drop-in replacement for a bare QgsMapLayerComboBox.
    """

    layerChanged = pyqtSignal(object)

    def __init__(self, kind, parent=None):
        super(LayerInput, self).__init__(parent)
        self.kind = kind
        self.combo = QgsMapLayerComboBox()
        flt = compat.layer_filter(kind)
        if flt is not None:
            try:
                self.combo.setFilters(flt)
            except (TypeError, ValueError):
                pass

        self.button = QToolButton()
        self.button.setText("...")
        self.button.setToolTip(
            "Load a layer from a file on disk and select it here.")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.combo, 1)
        row.addWidget(self.button)

        self.button.clicked.connect(self._browse)
        self.combo.layerChanged.connect(self.layerChanged.emit)

    def currentLayer(self):
        return self.combo.currentLayer()

    def setLayer(self, layer):
        self.combo.setLayer(layer)

    def setToolTip(self, text):
        self.combo.setToolTip(text)
        super(LayerInput, self).setToolTip(text)

    def _browse(self):
        is_raster = self.kind == "raster"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a raster layer" if is_raster else "Select a vector layer",
            "", RASTER_FILTER if is_raster else VECTOR_FILTER)
        if not path:
            return
        name = os.path.splitext(os.path.basename(path))[0]
        layer = (QgsRasterLayer(path, name) if is_raster
                 else QgsVectorLayer(path, name, "ogr"))
        if not layer.isValid():
            QMessageBox.warning(
                self, "EarthWorkQ",
                "'%s' could not be opened as a %s layer."
                % (os.path.basename(path), "raster" if is_raster else "vector"))
            return
        QgsProject.instance().addMapLayer(layer)
        self.combo.setLayer(layer)


# ----------------------------------------------------------------------
class EarthWorkQDialog(QDialog):

    runRequested = pyqtSignal()
    cancelRequested = pyqtSignal()

    def __init__(self, parent=None):
        super(EarthWorkQDialog, self).__init__(parent)
        self.setWindowTitle("EarthWorkQ - earthwork quantities")
        self.setWindowIcon(QIcon(os.path.join(PLUGIN_DIR, "icon.png")))
        self._enable_window_buttons()
        self._build()
        self._wire()
        self._refresh_engines()
        self._update_capability_notes()
        self._apply_screen_size()

    # ------------------------------------------------------------------
    # window behaviour
    # ------------------------------------------------------------------
    def _enable_window_buttons(self):
        """Give the dialog minimise, maximise and close buttons.

        A plain QDialog gets only a close button on most window managers, so it
        cannot be minimised out of the way or maximised to read a long log.
        """
        flags = self.windowFlags()
        for name in ("Window", "WindowSystemMenuHint",
                     "WindowMinimizeButtonHint", "WindowMaximizeButtonHint",
                     "WindowCloseButtonHint"):
            flag = _qt_flag(name)
            if flag is not None:
                flags |= flag
        self.setWindowFlags(flags)
        self.setSizeGripEnabled(True)

    def _available_geometry(self):
        screen = None
        try:
            screen = self.screen()
        except AttributeError:
            pass
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else None

    def _apply_screen_size(self):
        """Open at a size that always fits inside the work area.

        availableGeometry already excludes the task bar, so clamping to it is
        what keeps the button row on screen at high display scaling.
        """
        avail = self._available_geometry()
        if avail is None:
            self.resize(880, 660)
            return

        margin = 60
        max_w = max(480, avail.width() - margin)
        max_h = max(360, avail.height() - margin)

        self.setMinimumSize(min(560, max_w), min(400, max_h))
        self.resize(min(900, max_w), min(720, max_h))

        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        if frame.top() < avail.top():
            frame.moveTop(avail.top())
        if frame.bottom() > avail.bottom():
            frame.moveBottom(avail.bottom())
        if frame.left() < avail.left():
            frame.moveLeft(avail.left())
        self.move(frame.topLeft())

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(9, 9, 9, 9)
        outer.setSpacing(7)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        expanding = _size_policy("Expanding")
        self.tabs.setSizePolicy(expanding, expanding)
        outer.addWidget(self.tabs, 1)

        self.tabs.addTab(_scrollable(self._tab_elevation()), "1. Existing ground")
        self.tabs.addTab(_scrollable(self._tab_area()), "2. Area")
        self.tabs.addTab(_scrollable(self._tab_design()), "3. Design")
        self.tabs.addTab(_scrollable(self._tab_material()), "4. Material")
        self.tabs.addTab(_scrollable(self._tab_method()), "5. Method and checks")
        self.tabs.addTab(_scrollable(self._tab_outputs()), "6. Outputs")

        self.log_panel = self._log_panel()
        self.log_panel.setVisible(False)
        outer.addWidget(self.log_panel)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        row = QHBoxLayout()
        self.btn_toggle_log = QPushButton("Show log")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.setToolTip(
            "Show or hide the run log. It opens by itself when a calculation "
            "starts or when there is something to report.")
        self.btn_run = QPushButton("Calculate")
        self.btn_run.setDefault(True)
        self.btn_run.setToolTip("Run the calculation with the settings above.")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Stop the calculation currently running.")
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close the dialog. A running calculation is "
                                  "not cancelled by closing.")
        row.addWidget(self.btn_toggle_log)
        row.addStretch(1)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_close)
        outer.addLayout(row)

    # -- log -----------------------------------------------------------
    def _log_panel(self):
        box = QGroupBox("Run log")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 8)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(110)
        self.log.setStyleSheet(
            "font-family: Menlo, Consolas, 'DejaVu Sans Mono', monospace;"
            "font-size: 11px;")
        v.addWidget(self.log, 1)

        row = QHBoxLayout()
        self.btn_open_report = QPushButton("Open report")
        self.btn_open_report.setEnabled(False)
        self.btn_open_report.setToolTip(
            "Open the HTML calculation report in your browser.")
        self.btn_open_folder = QPushButton("Open output folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.setToolTip(
            "Open the folder the outputs were written to.")
        self.btn_save_log = QPushButton("Save log...")
        self.btn_save_log.setToolTip("Write the log above to a text file.")
        self.btn_clear_log = QPushButton("Clear")
        row.addWidget(self.btn_open_report)
        row.addWidget(self.btn_open_folder)
        row.addStretch(1)
        row.addWidget(self.btn_save_log)
        row.addWidget(self.btn_clear_log)
        v.addLayout(row)
        return box

    # -- tab 1 ---------------------------------------------------------
    def _tab_elevation(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_banner(
            "The surveyed existing ground surface. Choose whichever form your "
            "survey arrived in - a table of levels, a point layer, contour "
            "lines, or a DEM. All four are reduced to the same internal grid, "
            "so the choice affects accuracy rather than what you can do "
            "afterwards. Every input must be in a projected coordinate system "
            "in metres or feet: volumes cannot be computed from degrees."))

        form = QFormLayout()
        self.cmb_source_type = QComboBox()
        self.cmb_source_type.addItems([
            "Table of coordinates (CSV or Excel)",
            "Point layer",
            "Contour lines",
            "Digital elevation model",
        ])
        _add(form, "Elevation data from", self.cmb_source_type,
             "A table or point layer of surveyed levels gives the most "
             "reliable surface. Contours are the weakest input, because their "
             "vertical accuracy is limited to about half the contour interval.")
        v.addLayout(form)
        v.addWidget(_hline())

        self.stack_source = QStackedWidget()
        v.addWidget(self.stack_source)

        # --- table
        page = QWidget()
        f = QFormLayout(page)
        row = QHBoxLayout()
        self.txt_table_path = QLineEdit()
        self.txt_table_path.setPlaceholderText("Select a .csv, .txt or .xlsx file")
        self.btn_browse_table = QPushButton("Browse...")
        row.addWidget(self.txt_table_path, 1)
        row.addWidget(self.btn_browse_table)
        _add(f, "File", row,
             "A table with one row per surveyed level. The column headings are "
             "read as soon as the file is chosen, and matched automatically "
             "where they are recognisable.")
        self.cmb_field_east = QComboBox()
        self.cmb_field_north = QComboBox()
        self.cmb_field_elev = QComboBox()
        for c in (self.cmb_field_east, self.cmb_field_north, self.cmb_field_elev):
            c.setEditable(True)
        _add(f, "Easting column", self.cmb_field_east,
             "The X coordinate column. Matched automatically against easting, "
             "east and x.")
        _add(f, "Northing column", self.cmb_field_north,
             "The Y coordinate column. Matched automatically against northing, "
             "north and y.")
        _add(f, "Elevation column", self.cmb_field_elev,
             "The level column. Matched automatically against elevation, elev, "
             "z, level, RL and height.")
        try:
            from qgis.gui import QgsProjectionSelectionWidget
            self.crs_table = QgsProjectionSelectionWidget()
            _add(f, "Coordinate system", self.crs_table,
                 "A plain table carries no coordinate system of its own, so "
                 "one has to be stated here. It must be projected, not "
                 "geographic.")
        except ImportError:
            self.crs_table = None
        f.addRow(_hint(
            "Rows with a missing or non-numeric coordinate are discarded and "
            "counted in the report. Two points at the same place carrying "
            "different levels stop the run, because that is a transcription "
            "error rather than noise."))
        self.stack_source.addWidget(page)

        # --- point layer
        page = QWidget()
        f = QFormLayout(page)
        self.cmb_point_layer = LayerInput("point")
        _add(f, "Point layer", self.cmb_point_layer,
             "A layer of surveyed spot levels. Use the Browse button to load a "
             "shapefile or GeoPackage that is not already in the project.")
        self.cmb_point_field = QComboBox()
        _add(f, "Elevation attribute", self.cmb_point_field,
             "The numeric field holding the level of each point.")
        self.chk_use_geometry_z = _check(
            "Use Z from the geometry instead",
            "Take the level from the geometry's Z ordinate rather than from an "
            "attribute. Only works if the layer actually carries Z values.")
        f.addRow("", self.chk_use_geometry_z)
        f.addRow(_hint(
            "Points that differ sharply from their neighbours are flagged as "
            "possible level blunders. They are kept rather than deleted - the "
            "report tells you how many there are so you can review them."))
        self.stack_source.addWidget(page)

        # --- contours
        page = QWidget()
        f = QFormLayout(page)
        self.cmb_contour_layer = LayerInput("line")
        _add(f, "Contour layer", self.cmb_contour_layer,
             "Contour polylines, each carrying its elevation as an attribute.")
        self.cmb_contour_field = QComboBox()
        _add(f, "Elevation attribute", self.cmb_contour_field,
             "The numeric field holding the level of each contour line.")
        self.spn_densify = QDoubleSpinBox()
        self.spn_densify.setRange(0.0, 1000.0)
        self.spn_densify.setDecimals(2)
        self.spn_densify.setValue(0.0)
        self.spn_densify.setSpecialValueText("automatic")
        _add(f, "Densify vertices every", self.spn_densify,
             "Extra vertices are inserted along each contour before a surface "
             "is fitted. Without this, long straight segments between vertices "
             "produce visible terracing. Leave on automatic unless you have a "
             "reason not to.")
        self.chk_contour_topology = _check(
            "Check that contours do not cross (recommended)",
            "Contours at different elevations must never cross. If they do, "
            "the data is corrupt and any surface built from it will be wrong. "
            "On a very dense contour set this check is the slowest part of "
            "the run.", checked=True)
        f.addRow("", self.chk_contour_topology)
        f.addRow(_hint(
            "Vertical accuracy from contours is roughly half the contour "
            "interval. Integrated over the site that is usually the dominant "
            "error in the final quantity - larger than anything the choice of "
            "interpolation method contributes."))
        self.stack_source.addWidget(page)

        # --- DEM
        page = QWidget()
        f = QFormLayout(page)
        self.cmb_dem_layer = LayerInput("raster")
        _add(f, "DEM layer", self.cmb_dem_layer,
             "An existing digital elevation model of the site.")
        self.spn_dem_band = QSpinBox()
        self.spn_dem_band.setRange(1, 64)
        _add(f, "Band", self.spn_dem_band,
             "Which band holds the elevation. Almost always 1.")
        f.addRow(_hint(
            "A DEM is used directly - no interpolation beyond resampling onto "
            "the working grid. The cross-validation checks do not apply, "
            "because there are no survey points to hold back."))
        self.stack_source.addWidget(page)

        v.addStretch(1)
        return w

    # -- tab 2 ---------------------------------------------------------
    def _tab_area(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_banner(
            "The polygon or polygons the quantities are computed over. Results "
            "are reported for each polygon separately as well as for the site "
            "as a whole, so a layer of several platforms or phases can be run "
            "in one go."))

        f = QFormLayout()
        self.cmb_aoi = LayerInput("polygon")
        _add(f, "Area of interest", self.cmb_aoi,
             "A polygon layer in a projected coordinate system. Use Browse to "
             "load one that is not already in the project.")
        self.chk_selected_only = _check(
            "Use selected features only",
            "Restrict the calculation to the polygons currently selected in "
            "the layer rather than all of them.")
        f.addRow("", self.chk_selected_only)

        self.spn_subsample = QSpinBox()
        self.spn_subsample.setRange(1, 32)
        self.spn_subsample.setValue(8)
        _add(f, "Boundary subsampling", self.spn_subsample,
             "How finely each grid cell is divided when working out how much "
             "of it falls inside the polygon. 8 divides each cell into 64 "
             "parts. Higher is more accurate on the boundary and slower; 1 "
             "reduces to a plain centre-in-polygon test.")
        v.addLayout(f)
        v.addWidget(_hint(
            "Cells on the boundary are weighted by the fraction of their area "
            "inside the polygon rather than counted whole. A plain "
            "centre-in-polygon test introduces an error proportional to the "
            "perimeter of the site, which for a narrow strip - a road "
            "corridor, a channel, a haul road - is not a rounding "
            "difference."))
        v.addStretch(1)
        return w

    # -- tab 3 ---------------------------------------------------------
    def _tab_design(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_banner(
            "The formation surface the ground is being cut or filled to. Cut "
            "and fill are the difference between this surface and the existing "
            "ground, integrated over the area of interest."))

        f = QFormLayout()
        self.cmb_design_type = QComboBox()
        self.cmb_design_type.addItems([
            "Flat formation level",
            "Sloped formation plane",
            "Design DEM",
        ])
        _add(f, "Design surface", self.cmb_design_type,
             "A single level, a plane with drainage falls, or a full design "
             "surface produced elsewhere.")
        v.addLayout(f)

        self.stack_design = QStackedWidget()
        v.addWidget(self.stack_design)

        # flat
        page = QWidget()
        f = QFormLayout(page)
        self.spn_level = QDoubleSpinBox()
        self.spn_level.setRange(-11000.0, 11000.0)
        self.spn_level.setDecimals(3)
        _add(f, "Formation level", self.spn_level,
             "The level the whole platform is brought to, in the same datum as "
             "the survey.")
        self.chk_use_balance_level = _check(
            "Solve for the balancing level and use that instead",
            "Ignore the level above, compute the one at which the site "
            "balances, and report the quantities at that level. The level "
            "actually used is stated in the report.")
        f.addRow("", self.chk_use_balance_level)
        f.addRow(_hint(
            "The balancing level is where the material excavated is exactly "
            "what is needed to build the fill, after allowing for shrinkage. "
            "It is solved by bisection on the balance curve, which is monotone "
            "and so always converges."))
        self.stack_design.addWidget(page)

        # plane
        page = QWidget()
        f = QFormLayout(page)
        self.spn_plane_z = QDoubleSpinBox()
        self.spn_plane_z.setRange(-11000.0, 11000.0)
        self.spn_plane_z.setDecimals(3)
        _add(f, "Level at control point", self.spn_plane_z,
             "The formation level at the control point below. The plane passes "
             "through this level and falls away at the grades given.")
        row = QHBoxLayout()
        self.spn_plane_x = QDoubleSpinBox()
        self.spn_plane_y = QDoubleSpinBox()
        for s in (self.spn_plane_x, self.spn_plane_y):
            s.setRange(-1e9, 1e9)
            s.setDecimals(3)
        self.btn_plane_centroid = QPushButton("Use area centre")
        self.btn_plane_centroid.setToolTip(
            "Fill in the coordinates of the centre of the area of interest.")
        row.addWidget(self.spn_plane_x)
        row.addWidget(self.spn_plane_y)
        row.addWidget(self.btn_plane_centroid)
        _add(f, "Control point (E, N)", row,
             "The point at which the level above applies. Left at zero, the "
             "centre of the area of interest is used.")
        self.spn_grade_x = QDoubleSpinBox()
        self.spn_grade_y = QDoubleSpinBox()
        for s in (self.spn_grade_x, self.spn_grade_y):
            s.setRange(-100.0, 100.0)
            s.setDecimals(4)
            s.setSuffix(" %")
        _add(f, "Grade eastwards", self.spn_grade_x,
             "Rise per unit distance towards the east, as a percentage. 1.0% "
             "is 1 in 100. Negative falls towards the east.")
        _add(f, "Grade northwards", self.spn_grade_y,
             "Rise per unit distance towards the north, as a percentage. "
             "Negative falls towards the north.")
        f.addRow(_hint(
            "Real platforms are never flat - they fall to drainage. Modelling "
            "the fall matters: on a large platform a 1% crossfall moves a "
            "significant volume relative to a level surface at the same mean "
            "height."))
        self.stack_design.addWidget(page)

        # design DEM
        page = QWidget()
        f = QFormLayout(page)
        self.cmb_design_dem = LayerInput("raster")
        _add(f, "Design DEM", self.cmb_design_dem,
             "A raster of the design formation surface, in the same datum as "
             "the survey.")
        self.spn_design_offset = QDoubleSpinBox()
        self.spn_design_offset.setRange(-1000.0, 1000.0)
        self.spn_design_offset.setDecimals(3)
        _add(f, "Vertical offset", self.spn_design_offset,
             "Raise or lower the whole design surface bodily, keeping its "
             "shape. Useful for testing how sensitive the quantities are to "
             "the design level.")
        f.addRow(_hint(
            "With a design DEM the balance sweep raises and lowers the whole "
            "surface rather than varying a single level, so the curve is "
            "plotted against vertical offset rather than against level."))
        self.stack_design.addWidget(page)

        v.addStretch(1)
        return w

    # -- tab 4 ---------------------------------------------------------
    def _tab_material(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_banner(
            "Cut and fill volumes are not interchangeable. Compaction reduces "
            "the void ratio, so a cubic metre of placed fill needs more than a "
            "cubic metre of in-situ cut. The conversion follows from "
            "conservation of the volume of solids: shrinkage factor = "
            "(1 + e in-situ) / (1 + e compacted). Water content does not enter "
            "the volume balance at all - it is used only for conditioning "
            "water and haulage mass."))

        self.chk_apply_soil = _check(
            "Apply material properties (shrinkage, bulking, moisture)",
            "Turn this off to report raw cut and fill with no conversion "
            "between them. The balance level is then where raw cut equals raw "
            "fill, which normally leaves a site short of material.",
            checked=True)
        v.addWidget(self.chk_apply_soil)

        box = QGroupBox("Material")
        f = QFormLayout(box)
        self.cmb_material = QComboBox()
        self.cmb_material.addItems(list(MATERIAL_PRESETS.keys()))
        _add(f, "Preset", self.cmb_material,
             "Indicative starting values only. Always confirm them against "
             "site testing before issuing quantities.")

        self.cmb_input_mode = QComboBox()
        self.cmb_input_mode.addItems([
            "Void ratios",
            "Dry densities",
            "In-situ density and % of maximum dry density",
        ])
        _add(f, "Specify by", self.cmb_input_mode,
             "Enter whichever form your geotechnical data came in. All three "
             "resolve to the same pair of void ratios internally.")

        self.spn_e_insitu = QDoubleSpinBox()
        self.spn_e_insitu.setRange(0.01, 3.0)
        self.spn_e_insitu.setDecimals(3)
        self.spn_e_insitu.setSingleStep(0.01)
        _add(f, "In-situ void ratio", self.spn_e_insitu,
             "Void ratio of the undisturbed material in the cut - the volume "
             "of voids divided by the volume of solids.")

        self.spn_e_comp = QDoubleSpinBox()
        self.spn_e_comp.setRange(0.01, 3.0)
        self.spn_e_comp.setDecimals(3)
        self.spn_e_comp.setSingleStep(0.01)
        _add(f, "Compacted void ratio", self.spn_e_comp,
             "Void ratio once the material is placed and compacted. Lower than "
             "the in-situ value for soils; for rockfill it can be higher, "
             "which makes the shrinkage factor less than one.")

        self.spn_rho_insitu = QDoubleSpinBox()
        self.spn_rho_insitu.setRange(0.5, 3.0)
        self.spn_rho_insitu.setDecimals(3)
        self.spn_rho_insitu.setSuffix(" t/m3")
        _add(f, "In-situ dry density", self.spn_rho_insitu,
             "Dry density of the material in the ground, in tonnes per cubic "
             "metre. Watch the units - 1550 kg/m3 is 1.550 t/m3.")

        self.spn_rho_comp = QDoubleSpinBox()
        self.spn_rho_comp.setRange(0.5, 3.0)
        self.spn_rho_comp.setDecimals(3)
        self.spn_rho_comp.setSuffix(" t/m3")
        _add(f, "Compacted dry density", self.spn_rho_comp,
             "Dry density achieved in the compacted fill, in tonnes per cubic "
             "metre.")

        self.spn_mdd = QDoubleSpinBox()
        self.spn_mdd.setRange(0.5, 3.0)
        self.spn_mdd.setDecimals(3)
        self.spn_mdd.setSuffix(" t/m3")
        _add(f, "Maximum dry density (Proctor)", self.spn_mdd,
             "Maximum dry density from the Proctor compaction test. The "
             "specification is normally quoted as a percentage of this.")

        self.spn_compaction = QDoubleSpinBox()
        self.spn_compaction.setRange(50.0, 105.0)
        self.spn_compaction.setDecimals(1)
        self.spn_compaction.setValue(95.0)
        self.spn_compaction.setSuffix(" %")
        _add(f, "Compaction specified", self.spn_compaction,
             "The compaction requirement as a percentage of maximum dry "
             "density. 95% is a common earthworks specification.")

        self.spn_gs = QDoubleSpinBox()
        self.spn_gs.setRange(2.0, 3.5)
        self.spn_gs.setDecimals(3)
        self.spn_gs.setValue(2.65)
        _add(f, "Particle specific gravity", self.spn_gs,
             "Specific gravity of the soil solids. About 2.65 for most sands "
             "and 2.70 for clays. Affects the conversion between density and "
             "void ratio, and every mass output.")

        self.lbl_shrinkage = QLabel("-")
        self.lbl_shrinkage.setWordWrap(True)
        self.lbl_shrinkage.setStyleSheet("font-weight: 600;")
        _add(f, "Resulting shrinkage factor", self.lbl_shrinkage,
             "Recomputed as you type. This is the number that converts between "
             "cut and fill throughout the calculation.")
        v.addWidget(box)

        box = QGroupBox("Optional - haulage and moisture")
        f = QFormLayout(box)
        self.spn_e_loose = QDoubleSpinBox()
        self.spn_e_loose.setRange(0.0, 4.0)
        self.spn_e_loose.setDecimals(3)
        self.spn_e_loose.setSpecialValueText("not used")
        _add(f, "Loose (bulked) void ratio", self.spn_e_loose,
             "Void ratio of the material loose in a truck body. Material bulks "
             "when excavated, so this is higher than the in-situ value. Used "
             "to convert the surplus into loose volume for haulage.")
        self.spn_w_nat = QDoubleSpinBox()
        self.spn_w_nat.setRange(0.0, 100.0)
        self.spn_w_nat.setDecimals(1)
        self.spn_w_nat.setSuffix(" %")
        self.spn_w_nat.setSpecialValueText("not used")
        _add(f, "Natural water content", self.spn_w_nat,
             "Gravimetric water content of the material in the ground. Used "
             "for haul mass and to work out how much conditioning water is "
             "needed.")
        self.spn_w_target = QDoubleSpinBox()
        self.spn_w_target.setRange(0.0, 100.0)
        self.spn_w_target.setDecimals(1)
        self.spn_w_target.setSuffix(" %")
        self.spn_w_target.setSpecialValueText("not used")
        _add(f, "Target water content (OMC)", self.spn_w_target,
             "The water content the fill is to be placed at, normally the "
             "optimum from the Proctor test. If the natural content is well "
             "above this the material may be unsuitable, and you are warned.")
        self.spn_topsoil = QDoubleSpinBox()
        self.spn_topsoil.setRange(0.0, 5.0)
        self.spn_topsoil.setDecimals(3)
        _add(f, "Topsoil strip depth", self.spn_topsoil,
             "Depth of topsoil removed before earthworks begin. Its volume is "
             "reported separately and the earthworks are computed from the "
             "stripped surface downwards.")
        f.addRow(_hint(
            "Supplying both water contents also lets the plugin check that the "
            "degree of saturation they imply is physically possible. Numbers "
            "needing more water than there is void space to hold it are "
            "rejected rather than quietly used."))
        v.addWidget(box)
        v.addStretch(1)
        return w

    # -- tab 5 ---------------------------------------------------------
    def _tab_method(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_banner(
            "How the surface is built from the survey, how finely it is "
            "sampled, and which checks are run on the answer. The defaults are "
            "chosen to be defensible; the checks are what let you tell a "
            "reviewer why the number can be trusted."))

        box = QGroupBox("Surface interpolation")
        f = QFormLayout(box)
        self.cmb_engine = QComboBox()
        _add(f, "Engine", self.cmb_engine,
             "How elevations between the survey points are estimated. The list "
             "is filtered to the engines that suit the input type and are "
             "actually available in this QGIS installation.")
        self.lbl_engine_desc = QLabel("")
        self.lbl_engine_desc.setWordWrap(True)
        self.lbl_engine_desc.setStyleSheet(
            "color: palette(mid); font-size: 11px;")
        f.addRow("", self.lbl_engine_desc)

        self.spn_idw_power = QDoubleSpinBox()
        self.spn_idw_power.setRange(0.5, 6.0)
        self.spn_idw_power.setValue(2.0)
        self.spn_idw_power.setDecimals(1)
        _add(f, "IDW distance power", self.spn_idw_power,
             "How sharply influence falls off with distance. 2 is the usual "
             "choice; higher values make the surface more local and more "
             "bullseyed around individual points.")

        self.spn_neighbours = QSpinBox()
        self.spn_neighbours.setRange(3, 512)
        self.spn_neighbours.setValue(12)
        _add(f, "Neighbours used", self.spn_neighbours,
             "How many nearby survey points each estimate is based on. More "
             "gives a smoother surface and a slower run.")

        self.spn_smoothing = QDoubleSpinBox()
        self.spn_smoothing.setRange(0.0, 100.0)
        self.spn_smoothing.setDecimals(3)
        _add(f, "Spline smoothing", self.spn_smoothing,
             "Zero fits every point exactly. Raise it to let the surface pass "
             "near rather than through the data, which is what you want with "
             "noisy GNSS levels - honouring every point exactly means "
             "honouring every blunder exactly.")

        self.spn_tension = QDoubleSpinBox()
        self.spn_tension.setRange(1.0, 400.0)
        self.spn_tension.setValue(40.0)
        _add(f, "GRASS tension", self.spn_tension,
             "How tightly the spline is pulled towards the data. Raise it if "
             "the surface swings above or below the survey near steep breaks "
             "such as bank crests.")
        v.addWidget(box)

        box = QGroupBox("Working grid")
        f = QFormLayout(box)
        self.spn_cell = QDoubleSpinBox()
        self.spn_cell.setRange(0.0, 1000.0)
        self.spn_cell.setDecimals(3)
        self.spn_cell.setValue(0.0)
        self.spn_cell.setSpecialValueText("recommended from data density")
        _add(f, "Cell size", self.spn_cell,
             "The resolution everything is computed at. Left on automatic it "
             "is set to about half the mean spacing of the survey points.")
        self.lbl_recommended = QLabel("-")
        _add(f, "Recommended", self.lbl_recommended,
             "Calculated from the survey once the input is loaded.")
        f.addRow(_hint(
            "A finer grid does not add information the survey does not "
            "contain - it only smooths the appearance of the interpolation "
            "while making the run slower. Set a cell finer than recommended "
            "and the report says so."))
        v.addWidget(box)

        box = QGroupBox("Verification")
        f = QFormLayout(box)
        self.chk_cross_validate = _check(
            "Cross-validate every applicable interpolation engine",
            "Holds back a tenth of the survey points at a time and measures "
            "how well each engine predicts them, reporting an RMSE per engine. "
            "Turns the choice of interpolator into a measurement rather than "
            "an opinion.", checked=True)
        f.addRow("", self.chk_cross_validate)
        self.chk_convergence = _check(
            "Grid convergence test (recompute at finer cell sizes)",
            "Recomputes the quantities at half and a quarter of the chosen "
            "cell size. If the numbers are still moving, the cell size is too "
            "coarse to issue and the report says so. Roughly triples the run "
            "time.", checked=True)
        f.addRow("", self.chk_convergence)
        self.spn_balance_steps = QSpinBox()
        self.spn_balance_steps.setRange(20, 2000)
        self.spn_balance_steps.setValue(200)
        _add(f, "Balance curve steps", self.spn_balance_steps,
             "How many candidate levels are plotted on the balance curve. The "
             "sweep is effectively free, so raise it for a smoother chart. It "
             "does not affect the accuracy of the solved balance level, which "
             "is found by bisection.")
        f.addRow(_hint(
            "Where the design surface is planar, an independent exact "
            "integration over the survey triangles also runs automatically and "
            "is compared against the grid result. Disagreement of more than a "
            "couple of per cent means the cell size is too coarse."))
        v.addWidget(box)

        box = QGroupBox("What was found in this QGIS installation")
        vb = QVBoxLayout(box)
        self.lbl_capabilities = QLabel("")
        self.lbl_capabilities.setWordWrap(True)
        self.lbl_capabilities.setStyleSheet("font-size: 11px;")
        self.lbl_capabilities.setToolTip(
            "Optional packages change which engines and outputs are offered. "
            "Anything missing is hidden or skipped with a note rather than "
            "failing the run.")
        vb.addWidget(self.lbl_capabilities)
        v.addWidget(box)
        v.addStretch(1)
        return w

    # -- tab 6 ---------------------------------------------------------
    def _tab_outputs(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_banner(
            "Where the results are written and which products to produce. The "
            "HTML report is the deliverable: it records every input, "
            "assumption and check, so a reviewer can decide whether to trust "
            "the number without re-running the calculation."))

        f = QFormLayout()
        row = QHBoxLayout()
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setPlaceholderText("Folder for the output files")
        self.btn_browse_out = QPushButton("Browse...")
        row.addWidget(self.txt_output_dir, 1)
        row.addWidget(self.btn_browse_out)
        _add(f, "Output folder", row,
             "Everything is written here. Without a folder you get only the "
             "on-screen summary in the log.")
        self.txt_prefix = QLineEdit("earthworkq")
        _add(f, "File prefix", self.txt_prefix,
             "Prefixed to every output file name, so several runs can share a "
             "folder without overwriting each other.")
        self.txt_project = QLineEdit()
        self.txt_project.setPlaceholderText("Shown on the report")
        _add(f, "Project name", self.txt_project,
             "Appears in the report heading.")
        self.txt_prepared_by = QLineEdit()
        _add(f, "Prepared by", self.txt_prepared_by,
             "Appears under the report heading alongside the date and time.")
        v.addLayout(f)

        box = QGroupBox("Products")
        f = QFormLayout(box)
        self.chk_out_raster = _check(
            "Cut/fill depth raster, existing and design surfaces",
            "A signed depth raster - negative is cut, positive is fill - "
            "styled with a diverging ramp centred on zero, plus the two "
            "surfaces it was derived from.", checked=True)
        self.chk_out_zones = _check(
            "Cut / balanced / fill zone polygons",
            "Polygons of where the site is in cut, in fill, and within the "
            "no-change band, with the area of each.", checked=True)
        self.chk_out_contours = _check(
            "Depth contours of the difference surface",
            "Contours of cut and fill depth, useful for setting out on site.")
        self.chk_out_chart = _check(
            "Balance curve and area-elevation charts",
            "Cut, fill and net plotted against formation level with the "
            "balance point marked, plus the site's area-elevation curve. Needs "
            "matplotlib.", checked=True)
        self.chk_out_csv = _check(
            "Quantities and balance curve as CSV",
            "Per-polygon quantities and the full balance curve as CSV, for "
            "taking into a spreadsheet or a bill of quantities.", checked=True)
        self.chk_out_report = _check(
            "HTML calculation report",
            "The full record of inputs, assumptions, quantities and "
            "verification checks, with the charts embedded.", checked=True)
        self.chk_write_back = _check(
            "Write quantities back onto the area layer",
            "Adds cut, fill, net and area attributes to each polygon in the "
            "area of interest layer. Skipped with a note if the layer does not "
            "accept new fields.", checked=True)
        self.chk_add_layers = _check(
            "Add the outputs to the project",
            "Load the rasters and polygons into the current QGIS project as "
            "they are created.", checked=True)
        for c in (self.chk_out_raster, self.chk_out_zones, self.chk_out_contours,
                  self.chk_out_chart, self.chk_out_csv, self.chk_out_report,
                  self.chk_write_back, self.chk_add_layers):
            f.addRow("", c)

        self.spn_zone_tol = QDoubleSpinBox()
        self.spn_zone_tol.setRange(0.0, 5.0)
        self.spn_zone_tol.setDecimals(3)
        self.spn_zone_tol.setValue(0.05)
        _add(f, "No-change band (+/-)", self.spn_zone_tol,
             "Depths within this band of zero are mapped as balanced rather "
             "than as cut or fill. Affects the zone polygons only, never the "
             "quantities.")
        self.spn_contour_interval = QDoubleSpinBox()
        self.spn_contour_interval.setRange(0.0, 100.0)
        self.spn_contour_interval.setDecimals(2)
        self.spn_contour_interval.setSpecialValueText("automatic")
        _add(f, "Depth contour interval", self.spn_contour_interval,
             "Vertical interval of the cut/fill depth contours. Automatic "
             "divides the observed depth range into about ten steps.")
        v.addWidget(box)

        lbl = QLabel("Report notes")
        lbl.setToolTip("Free text appended to the end of the report.")
        v.addWidget(lbl)
        self.txt_notes = QTextEdit()
        self.txt_notes.setPlaceholderText(
            "Notes to appear on the report - assumptions, survey reference, "
            "drawing numbers")
        self.txt_notes.setMaximumHeight(84)
        self.txt_notes.setToolTip(
            "Anything a reviewer should know that the plugin cannot work out "
            "for itself: the survey reference, the drawing revision, "
            "assumptions agreed with the client.")
        v.addWidget(self.txt_notes)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def _wire(self):
        self.cmb_source_type.currentIndexChanged.connect(
            self.stack_source.setCurrentIndex)
        self.cmb_source_type.currentIndexChanged.connect(self._refresh_engines)
        self.cmb_design_type.currentIndexChanged.connect(
            self.stack_design.setCurrentIndex)
        self.btn_browse_table.clicked.connect(self._browse_table)
        self.btn_browse_out.clicked.connect(self._browse_output)
        self.cmb_point_layer.layerChanged.connect(
            lambda lyr: self._fill_fields(lyr, self.cmb_point_field))
        self.cmb_contour_layer.layerChanged.connect(
            lambda lyr: self._fill_fields(lyr, self.cmb_contour_field))
        self.cmb_engine.currentIndexChanged.connect(self._engine_changed)
        self.cmb_material.currentTextChanged.connect(self._apply_preset)
        self.cmb_input_mode.currentIndexChanged.connect(self._material_mode_changed)
        self.chk_apply_soil.toggled.connect(self._update_shrinkage)
        for s in (self.spn_e_insitu, self.spn_e_comp, self.spn_rho_insitu,
                  self.spn_rho_comp, self.spn_mdd, self.spn_compaction,
                  self.spn_gs):
            s.valueChanged.connect(self._update_shrinkage)
        self.txt_output_dir.textChanged.connect(
            lambda t: self.btn_open_folder.setEnabled(os.path.isdir(t.strip() or "\0")))
        self.btn_run.clicked.connect(self.runRequested.emit)
        self.btn_cancel.clicked.connect(self.cancelRequested.emit)
        self.btn_close.clicked.connect(self.reject)
        self.btn_clear_log.clicked.connect(self.log.clear)
        self.btn_save_log.clicked.connect(self._save_log)
        self.btn_toggle_log.toggled.connect(self._toggle_log)
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        self.btn_plane_centroid.clicked.connect(self._use_centroid)

        self._fill_fields(self.cmb_point_layer.currentLayer(), self.cmb_point_field)
        self._fill_fields(self.cmb_contour_layer.currentLayer(),
                          self.cmb_contour_field)
        self._apply_preset(self.cmb_material.currentText())
        self._material_mode_changed(0)

    # ------------------------------------------------------------------
    # log panel
    # ------------------------------------------------------------------
    def _toggle_log(self, shown):
        self.log_panel.setVisible(shown)
        self.btn_toggle_log.setText("Hide log" if shown else "Show log")

    def show_log(self):
        if not self.btn_toggle_log.isChecked():
            self.btn_toggle_log.setChecked(True)

    def _open_output_folder(self):
        path = self.txt_output_dir.text().strip()
        if path and os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # ------------------------------------------------------------------
    # browsing
    # ------------------------------------------------------------------
    def _browse_table(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an elevation table", "",
            "Tables (*.csv *.txt *.tsv *.xlsx *.xlsm);;All files (*)")
        if not path:
            return
        self.txt_table_path.setText(path)
        try:
            from ..core.sources import table_columns
            cols = table_columns(path)
        except Exception as exc:
            self.warn("Could not read the file header: %s" % exc)
            return
        for cmb, guesses in ((self.cmb_field_east, ("easting", "east", "x")),
                             (self.cmb_field_north, ("northing", "north", "y")),
                             (self.cmb_field_elev, ("elevation", "elev", "z",
                                                    "level", "rl", "height"))):
            cmb.clear()
            cmb.addItems(cols)
            lower = [c.strip().lower() for c in cols]
            for g in guesses:
                if g in lower:
                    cmb.setCurrentIndex(lower.index(g))
                    break

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select an output folder")
        if path:
            self.txt_output_dir.setText(path)

    def _fill_fields(self, layer, combo):
        combo.clear()
        if layer is None:
            return
        names = [f.name() for f in layer.fields()]
        combo.addItems(names)
        for guess in ("elev", "elevation", "z", "level", "rl", "height", "contour"):
            for i, n in enumerate(names):
                if n.strip().lower() == guess:
                    combo.setCurrentIndex(i)
                    return

    def _use_centroid(self):
        layer = self.cmb_aoi.currentLayer()
        if layer is None:
            self.warn("Choose an area of interest layer first.")
            return
        e = layer.extent()
        self.spn_plane_x.setValue(e.center().x())
        self.spn_plane_y.setValue(e.center().y())

    # ------------------------------------------------------------------
    # engines
    # ------------------------------------------------------------------
    def _refresh_engines(self):
        kind = ("contours" if self.cmb_source_type.currentIndex() == SOURCE_CONTOURS
                else "points")
        is_dem = self.cmb_source_type.currentIndex() == SOURCE_DEM
        self.cmb_engine.blockSignals(True)
        self.cmb_engine.clear()
        if is_dem:
            self.cmb_engine.addItem("Not applicable - DEM used directly", None)
            self.cmb_engine.setEnabled(False)
        else:
            self.cmb_engine.setEnabled(True)
            has_scipy = interp.scipy_available()
            has_grass = interp.grass_available()
            for cls in interp.engines_for(kind):
                if cls.id.startswith("grass") and not has_grass:
                    continue
                if not getattr(cls, "produces_raster", False) and not has_scipy:
                    continue
                self.cmb_engine.addItem(cls.display_name, cls.id)
            default = interp.DEFAULT_ENGINE.get(kind)
            idx = self.cmb_engine.findData(default)
            if idx < 0:
                idx = self.cmb_engine.findData(interp.FALLBACK_ENGINE.get(kind))
            self.cmb_engine.setCurrentIndex(max(idx, 0))
        self.cmb_engine.blockSignals(False)
        self._engine_changed()

    def _engine_changed(self):
        eid = self.cmb_engine.currentData()
        if not eid:
            self.lbl_engine_desc.setText("")
            return
        try:
            cls = interp.engine_by_id(eid)
        except Exception:
            return
        self.lbl_engine_desc.setText(cls.description)
        self.cmb_engine.setToolTip(cls.description)
        self.spn_idw_power.setEnabled(eid == "idw")
        self.spn_neighbours.setEnabled(eid in ("idw", "spline_rbf"))
        self.spn_smoothing.setEnabled(eid in ("spline_rbf", "grass_rst"))
        self.spn_tension.setEnabled(eid == "grass_rst")

    def _update_capability_notes(self):
        rows = [("SciPy",
                 "available" if interp.scipy_available()
                 else "not found - only the QGIS TIN engine is offered"),
                ("GRASS provider",
                 "available" if interp.grass_available() else "not found")]
        try:
            import matplotlib  # noqa: F401
            rows.append(("Matplotlib", "available"))
        except ImportError:
            rows.append(("Matplotlib",
                         "not found - charts will be written as CSV only"))
        try:
            import openpyxl  # noqa: F401
            rows.append(("openpyxl", "available"))
        except ImportError:
            rows.append(("openpyxl",
                         "not found - .xlsx input unavailable, use CSV"))
        self.lbl_capabilities.setText(
            "<br>".join("<b>%s</b>: %s" % r for r in rows))

    # ------------------------------------------------------------------
    # material
    # ------------------------------------------------------------------
    def _apply_preset(self, name):
        preset = MATERIAL_PRESETS.get(name)
        if not preset:
            return
        self.spn_e_insitu.setValue(preset["e_insitu"])
        self.spn_e_comp.setValue(preset["e_compacted"])
        self.spn_e_loose.setValue(preset.get("e_loose", 0.0) or 0.0)
        self.spn_gs.setValue(preset.get("gs", 2.65))
        self._update_shrinkage()

    def _material_mode_changed(self, index):
        by_void = index == 0
        by_density = index == 1
        by_spec = index == 2
        self.spn_e_insitu.setEnabled(by_void)
        self.spn_e_comp.setEnabled(by_void)
        self.spn_rho_insitu.setEnabled(by_density or by_spec)
        self.spn_rho_comp.setEnabled(by_density)
        self.spn_mdd.setEnabled(by_spec)
        self.spn_compaction.setEnabled(by_spec)
        self._update_shrinkage()

    def _update_shrinkage(self):
        try:
            soil = self.build_soil()
        except Exception as exc:
            self.lbl_shrinkage.setStyleSheet("color: #a4161a; font-size: 11px;")
            self.lbl_shrinkage.setText(str(exc))
            return
        if soil is None:
            self.lbl_shrinkage.setStyleSheet("color: palette(mid);")
            self.lbl_shrinkage.setText("not applied - raw cut and fill reported")
            return
        self.lbl_shrinkage.setStyleSheet("font-weight: 600;")
        self.lbl_shrinkage.setText(
            "%.4f  -  %.0f m3 of cut makes 100 m3 of compacted fill"
            % (soil.shrinkage_factor, soil.shrinkage_factor * 100))

    def build_soil(self):
        """Assemble a SoilModel from the material tab, or None."""
        from ..core.soil import SoilModel
        if not self.chk_apply_soil.isChecked():
            return None
        mode = self.cmb_input_mode.currentIndex()
        extras = dict(
            gs=self.spn_gs.value(),
            e_loose=(self.spn_e_loose.value() or None),
            w_natural=(self.spn_w_nat.value() / 100.0
                       if self.spn_w_nat.value() else None),
            w_target=(self.spn_w_target.value() / 100.0
                      if self.spn_w_target.value() else None),
            topsoil_strip=self.spn_topsoil.value(),
            name=self.cmb_material.currentText(),
        )
        if mode == 0:
            return SoilModel(self.spn_e_insitu.value(),
                             self.spn_e_comp.value(), **extras)
        if mode == 1:
            return SoilModel.from_densities(self.spn_rho_insitu.value(),
                                            self.spn_rho_comp.value(), **extras)
        return SoilModel.from_compaction_spec(
            self.spn_rho_insitu.value(), self.spn_mdd.value(),
            self.spn_compaction.value() / 100.0, **extras)

    def engine_params(self):
        eid = self.cmb_engine.currentData()
        if eid == "idw":
            return dict(power=self.spn_idw_power.value(),
                        neighbours=self.spn_neighbours.value())
        if eid == "spline_rbf":
            return dict(smoothing=self.spn_smoothing.value(),
                        neighbours=self.spn_neighbours.value())
        if eid == "grass_rst":
            return dict(tension=self.spn_tension.value(),
                        smooth=self.spn_smoothing.value())
        return {}

    # ------------------------------------------------------------------
    # messages and run state
    # ------------------------------------------------------------------
    def append_log(self, text):
        self.show_log()
        self.log.append(str(text))
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def warn(self, message):
        QMessageBox.warning(self, "EarthWorkQ", message)

    def error(self, message):
        self.show_log()
        QMessageBox.critical(self, "EarthWorkQ", message)

    def info(self, message):
        QMessageBox.information(self, "EarthWorkQ", message)

    def set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.tabs.setEnabled(not running)
        if running:
            self.progress.setVisible(True)   # stays visible so the final
            self.show_log()                  # 100% is actually seen
        else:
            self.btn_open_folder.setEnabled(
                os.path.isdir(self.txt_output_dir.text().strip() or "\0"))

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save log", "earthworkq.log", "Log files (*.log *.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.log.toPlainText())
