# -*- coding: utf-8 -*-
"""EarthWorkQ - earthwork cut and fill quantities for QGIS."""


def classFactory(iface):
    from .earthworkq import EarthWorkQPlugin
    return EarthWorkQPlugin(iface)
