# -*- coding: utf-8 -*-
"""Processing provider for EarthWorkQ.

Exposing the engine as Processing algorithms means the same code that the
dialog runs is available in the model builder, in batch mode and from PyQGIS,
and can be tested headlessly.
"""

from __future__ import annotations

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithms import (
    BalanceLevelAlgorithm,
    EarthworkQuantitiesAlgorithm,
    SurfaceFromSurveyAlgorithm,
)

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EarthWorkQProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        for alg in (EarthworkQuantitiesAlgorithm(),
                    BalanceLevelAlgorithm(),
                    SurfaceFromSurveyAlgorithm()):
            self.addAlgorithm(alg)

    def id(self):
        return "earthworkq"

    def name(self):
        return "EarthWorkQ"

    def longName(self):
        return "EarthWorkQ - earthwork cut and fill quantities"

    def icon(self):
        return QIcon(os.path.join(PLUGIN_DIR, "icon.png"))
