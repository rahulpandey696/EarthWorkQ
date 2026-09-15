# -*- coding: utf-8 -*-
"""
QGIS 3 / QGIS 4 compatibility shims.

Between QGIS 3 and QGIS 4 a number of unscoped enumerators were moved into
scoped enums under ``Qgis``, some classes were removed outright, and PyQt6
dropped the ``QVariant.Double`` style type constants that ``QgsField`` used to
take.  Everything that differs is resolved here once, so the rest of the plugin
can be written against a single stable set of names.

Nothing in this module raises at import time: each lookup falls back through
the alternatives and only fails when the value is actually used, which keeps a
single relocated enumerator from preventing the whole plugin from loading.
"""

from __future__ import annotations

from qgis.core import QgsField, QgsTask, QgsWkbTypes

try:
    from qgis.core import Qgis
except ImportError:                                   # pragma: no cover
    Qgis = None


def _first(*candidates):
    """Return the first candidate that resolves to something other than None."""
    for get in candidates:
        try:
            value = get()
        except (AttributeError, ImportError, TypeError, NameError):
            continue
        if value is not None:
            return value
    return None


# ----------------------------------------------------------------------
# fields
# ----------------------------------------------------------------------
def make_field(name, kind="double"):
    """Build a QgsField without caring which type constants this build uses."""
    # QGIS 3.38+ / Qt6 prefer QMetaType
    try:
        from qgis.PyQt.QtCore import QMetaType
        mapping = {"double": QMetaType.Type.Double,
                   "int": QMetaType.Type.Int,
                   "string": QMetaType.Type.QString}
        return QgsField(name, mapping[kind])
    except Exception:
        pass
    try:
        from qgis.PyQt.QtCore import QVariant
        mapping = {"double": QVariant.Double,
                   "int": QVariant.Int,
                   "string": QVariant.String}
        return QgsField(name, mapping[kind])
    except Exception:
        pass
    return QgsField(name)


# ----------------------------------------------------------------------
# geometry types
# ----------------------------------------------------------------------
GEOM_POINT = _first(lambda: Qgis.GeometryType.Point,
                    lambda: QgsWkbTypes.PointGeometry)
GEOM_LINE = _first(lambda: Qgis.GeometryType.Line,
                   lambda: QgsWkbTypes.LineGeometry)
GEOM_POLYGON = _first(lambda: Qgis.GeometryType.Polygon,
                      lambda: QgsWkbTypes.PolygonGeometry)


def geometry_type(layer):
    try:
        return layer.geometryType()
    except AttributeError:
        return QgsWkbTypes.geometryType(layer.wkbType())


def is_point_layer(layer):
    return geometry_type(layer) == GEOM_POINT


def is_line_layer(layer):
    return geometry_type(layer) == GEOM_LINE


def has_z(layer):
    return QgsWkbTypes.hasZ(layer.wkbType())


# ----------------------------------------------------------------------
# map layer combo box filters
# ----------------------------------------------------------------------
def layer_filter(which):
    """Filter flag for QgsMapLayerComboBox.setFilters()."""
    names = {"point": "PointLayer", "line": "LineLayer",
             "polygon": "PolygonLayer", "raster": "RasterLayer"}
    attr = names[which]
    value = _first(lambda: getattr(Qgis.LayerFilter, attr))
    if value is not None:
        return value
    try:
        from qgis.core import QgsMapLayerProxyModel
    except ImportError:
        return None
    return _first(lambda: getattr(QgsMapLayerProxyModel.Filter, attr),
                  lambda: getattr(QgsMapLayerProxyModel, attr))


# ----------------------------------------------------------------------
# raster shading
# ----------------------------------------------------------------------
def colour_ramp_interpolated():
    from qgis.core import QgsColorRampShader
    return _first(lambda: Qgis.ShaderInterpolationMethod.Linear,
                  lambda: QgsColorRampShader.Type.Interpolated,
                  lambda: QgsColorRampShader.Interpolated)


# ----------------------------------------------------------------------
# vector provider capabilities
# ----------------------------------------------------------------------
def provider_can(provider, what):
    """``what`` is 'AddAttributes' or 'ChangeAttributeValues'."""
    try:
        caps = provider.capabilities()
    except Exception:
        return True
    flag = _first(lambda: getattr(Qgis.VectorProviderCapability, what))
    if flag is None:
        try:
            from qgis.core import QgsVectorDataProvider
            flag = _first(lambda: getattr(QgsVectorDataProvider.Capability, what),
                          lambda: getattr(QgsVectorDataProvider, what))
        except ImportError:
            return True
    if flag is None:
        return True
    try:
        return bool(caps & flag)
    except TypeError:
        return True


# ----------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------
TASK_CAN_CANCEL = _first(lambda: QgsTask.Flag.CanCancel,
                         lambda: QgsTask.CanCancel)


# ----------------------------------------------------------------------
# processing
# ----------------------------------------------------------------------
def processing_source_type(which):
    """``which`` is 'VectorPoint', 'VectorLine' or 'VectorPolygon'."""
    from qgis.core import QgsProcessing
    return _first(lambda: getattr(Qgis.ProcessingSourceType, which),
                  lambda: getattr(QgsProcessing.SourceType, "Type" + which),
                  lambda: getattr(QgsProcessing, "Type" + which))


def processing_field_type(which="Numeric"):
    from qgis.core import QgsProcessingParameterField
    return _first(lambda: getattr(QgsProcessingParameterField.DataType, which),
                  lambda: getattr(QgsProcessingParameterField, which))


def processing_number_type(which="Double"):
    from qgis.core import QgsProcessingParameterNumber
    return _first(lambda: getattr(QgsProcessingParameterNumber.Type, which),
                  lambda: getattr(QgsProcessingParameterNumber, which))


# ----------------------------------------------------------------------
# message levels
# ----------------------------------------------------------------------
LEVEL_INFO = _first(lambda: Qgis.MessageLevel.Info, lambda: Qgis.Info)
LEVEL_WARNING = _first(lambda: Qgis.MessageLevel.Warning, lambda: Qgis.Warning)
LEVEL_CRITICAL = _first(lambda: Qgis.MessageLevel.Critical, lambda: Qgis.Critical)
