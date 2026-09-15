# -*- coding: utf-8 -*-
"""
Soil / material model for EarthWorkQ.

Volume conversion between in-situ cut and compacted fill is governed by
conservation of the volume of soil SOLIDS, not by conservation of total
volume.  For a soil of void ratio e:

    V_solids = V_total / (1 + e)

Balancing solids between an excavation and a compacted embankment gives:

    V_cut / (1 + e_insitu) = V_fill / (1 + e_compacted)

    =>  V_cut = V_fill * (1 + e_insitu) / (1 + e_compacted)

The multiplier is the SHRINKAGE FACTOR.  Because compaction reduces the
void ratio, the factor is normally greater than 1.0 (typically 1.10-1.30),
i.e. more cubic metres must be excavated than are placed.

Water content does NOT enter the volume balance - water is added or driven
off during conditioning and compaction, so it is not conserved.  It is used
only for (a) conditioning water quantities, (b) haulage masses and
(c) validating that the supplied parameters describe a physically possible
soil (degree of saturation must not exceed 1).
"""

from __future__ import annotations

import math

RHO_WATER = 1.0  # t/m3


class SoilParameterError(ValueError):
    """Raised when the supplied material parameters are physically impossible."""


def void_ratio_from_dry_density(rho_d, particle_density_gs):
    """e = Gs * rho_w / rho_d - 1   (rho_d in t/m3)."""
    if rho_d is None or rho_d <= 0:
        raise SoilParameterError("Dry density must be greater than zero.")
    return (particle_density_gs * RHO_WATER / float(rho_d)) - 1.0


def dry_density_from_void_ratio(e, particle_density_gs):
    return particle_density_gs * RHO_WATER / (1.0 + e)


class SoilModel(object):
    """Material properties controlling the cut/fill volume conversion.

    Parameters
    ----------
    e_insitu : float
        Void ratio of the undisturbed material in the cut.
    e_compacted : float
        Void ratio of the material once placed and compacted in the fill.
    e_loose : float or None
        Void ratio of the loose (bulked) material in a truck body.  Used only
        for haulage estimates.
    gs : float
        Particle specific gravity (2.60-2.75 for most soils).
    w_natural, w_target : float or None
        Gravimetric water contents as decimals (0.12 = 12%).  Optional.
    topsoil_strip : float
        Depth of topsoil stripped before earthworks (m).  Reported separately
        and removed from the earthworks surface.
    """

    def __init__(self, e_insitu, e_compacted, gs=2.65, e_loose=None,
                 w_natural=None, w_target=None, topsoil_strip=0.0,
                 name="Unnamed material"):
        self.name = name
        self.e_insitu = float(e_insitu)
        self.e_compacted = float(e_compacted)
        self.e_loose = None if e_loose is None else float(e_loose)
        self.gs = float(gs)
        self.w_natural = None if w_natural is None else float(w_natural)
        self.w_target = None if w_target is None else float(w_target)
        self.topsoil_strip = max(0.0, float(topsoil_strip or 0.0))
        self.validate()

    # ------------------------------------------------------------------
    # alternative constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_densities(cls, rho_d_insitu, rho_d_compacted, gs=2.65, **kwargs):
        return cls(void_ratio_from_dry_density(rho_d_insitu, gs),
                   void_ratio_from_dry_density(rho_d_compacted, gs),
                   gs=gs, **kwargs)

    @classmethod
    def from_compaction_spec(cls, rho_d_insitu, max_dry_density,
                             compaction_fraction, gs=2.65, **kwargs):
        """Compacted state defined as a percentage of Proctor MDD."""
        if not (0.5 <= compaction_fraction <= 1.05):
            raise SoilParameterError(
                "Compaction specification must be between 50% and 105% of MDD.")
        rho_d_comp = float(max_dry_density) * float(compaction_fraction)
        return cls.from_densities(rho_d_insitu, rho_d_comp, gs=gs, **kwargs)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self):
        if self.gs <= 1.0 or self.gs > 5.0:
            raise SoilParameterError(
                "Particle specific gravity of %.3f is outside any credible "
                "range (expected roughly 2.4 - 3.0)." % self.gs)
        for label, e in (("in-situ", self.e_insitu),
                         ("compacted", self.e_compacted)):
            if e <= 0.0:
                raise SoilParameterError(
                    "The %s void ratio must be greater than zero." % label)
            if e > 3.0:
                raise SoilParameterError(
                    "The %s void ratio of %.2f is implausibly high; check "
                    "whether a dry density was entered in kg/m3 instead of "
                    "t/m3." % (label, e))
        if self.e_loose is not None and self.e_loose < self.e_insitu:
            raise SoilParameterError(
                "Loose void ratio cannot be lower than the in-situ void ratio "
                "- material bulks when it is excavated.")
        # degree of saturation check: S = w * Gs / e
        for label, w, e in (("natural", self.w_natural, self.e_insitu),
                            ("target", self.w_target, self.e_compacted)):
            if w is None:
                continue
            if w < 0:
                raise SoilParameterError("Water content cannot be negative.")
            s = self.degree_of_saturation(w, e)
            if s > 1.0001:
                raise SoilParameterError(
                    "The %s water content of %.1f%% implies a degree of "
                    "saturation of %.0f%%, which is impossible for a void "
                    "ratio of %.3f. Check the water content or the density."
                    % (label, w * 100.0, s * 100.0, e))

    def degree_of_saturation(self, w, e):
        return (w * self.gs) / e if e > 0 else float("inf")

    # ------------------------------------------------------------------
    # volume conversion
    # ------------------------------------------------------------------
    @property
    def shrinkage_factor(self):
        """Cubic metres of in-situ cut required per cubic metre of placed fill."""
        return (1.0 + self.e_insitu) / (1.0 + self.e_compacted)

    @property
    def bulking_factor(self):
        """Loose volume per unit in-situ volume (haulage)."""
        if self.e_loose is None:
            return None
        return (1.0 + self.e_loose) / (1.0 + self.e_insitu)

    def fill_from_cut(self, v_cut):
        """Compacted fill obtainable from a given in-situ cut volume."""
        return v_cut / self.shrinkage_factor

    def cut_for_fill(self, v_fill):
        """In-situ cut required to produce a given compacted fill volume."""
        return v_fill * self.shrinkage_factor

    # ------------------------------------------------------------------
    # mass / water
    # ------------------------------------------------------------------
    def solids_volume_from_cut(self, v_cut):
        return v_cut / (1.0 + self.e_insitu)

    def solids_mass_from_cut(self, v_cut):
        """Tonnes of dry solids in a given in-situ cut volume."""
        return self.solids_volume_from_cut(v_cut) * self.gs * RHO_WATER

    def loose_volume_from_cut(self, v_cut):
        bf = self.bulking_factor
        return None if bf is None else v_cut * bf

    def haul_mass_from_cut(self, v_cut):
        """Tonnes actually carried, including natural moisture."""
        if self.w_natural is None:
            return None
        return self.solids_mass_from_cut(v_cut) * (1.0 + self.w_natural)

    def conditioning_water(self, v_fill_placed):
        """Cubic metres of water to add (positive) or remove (negative).

        Referenced to the mass of solids that ends up in the compacted fill.
        """
        if self.w_natural is None or self.w_target is None:
            return None
        solids_mass = v_fill_placed / (1.0 + self.e_compacted) * self.gs * RHO_WATER
        return solids_mass * (self.w_target - self.w_natural) / RHO_WATER

    def moisture_warning(self):
        if self.w_natural is None or self.w_target is None:
            return None
        delta = self.w_natural - self.w_target
        if delta > 0.04:
            return ("Natural water content is %.1f%% above the target. The "
                    "material is likely to be too wet to compact without "
                    "drying or stabilisation, and may be classed unsuitable."
                    % (delta * 100.0))
        if delta < -0.04:
            return ("Natural water content is %.1f%% below the target. Water "
                    "will have to be added on site; the conditioning quantity "
                    "is reported below." % (-delta * 100.0))
        return None

    # ------------------------------------------------------------------
    def summary(self):
        rows = [
            ("Material", self.name),
            ("In-situ void ratio", "%.3f" % self.e_insitu),
            ("Compacted void ratio", "%.3f" % self.e_compacted),
            ("Particle specific gravity", "%.2f" % self.gs),
            ("In-situ dry density",
             "%.3f t/m3" % dry_density_from_void_ratio(self.e_insitu, self.gs)),
            ("Compacted dry density",
             "%.3f t/m3" % dry_density_from_void_ratio(self.e_compacted, self.gs)),
            ("Shrinkage factor (cut per unit fill)",
             "%.4f" % self.shrinkage_factor),
        ]
        if self.e_loose is not None:
            rows.append(("Bulking factor (loose per unit in-situ)",
                         "%.4f" % self.bulking_factor))
        if self.w_natural is not None:
            rows.append(("Natural water content", "%.1f %%" % (self.w_natural * 100)))
        if self.w_target is not None:
            rows.append(("Target water content", "%.1f %%" % (self.w_target * 100)))
        if self.topsoil_strip > 0:
            rows.append(("Topsoil strip depth", "%.3f m" % self.topsoil_strip))
        return rows

    def as_dict(self):
        return {
            "name": self.name,
            "e_insitu": self.e_insitu,
            "e_compacted": self.e_compacted,
            "e_loose": self.e_loose,
            "gs": self.gs,
            "w_natural": self.w_natural,
            "w_target": self.w_target,
            "topsoil_strip": self.topsoil_strip,
            "shrinkage_factor": self.shrinkage_factor,
        }


#: Indicative starting values only - always confirm against site testing.
MATERIAL_PRESETS = {
    "Uniform sand (loose in-situ)":      dict(e_insitu=0.70, e_compacted=0.55, e_loose=0.85, gs=2.65),
    "Well graded sand and gravel":       dict(e_insitu=0.55, e_compacted=0.40, e_loose=0.75, gs=2.68),
    "Firm clay":                         dict(e_insitu=0.80, e_compacted=0.62, e_loose=1.05, gs=2.70),
    "Stiff clay":                        dict(e_insitu=0.65, e_compacted=0.52, e_loose=0.90, gs=2.70),
    "Silty soil":                        dict(e_insitu=0.75, e_compacted=0.58, e_loose=0.95, gs=2.67),
    "Weathered rock / rockfill":         dict(e_insitu=0.30, e_compacted=0.35, e_loose=0.70, gs=2.70),
    "Custom":                            dict(e_insitu=0.70, e_compacted=0.55, e_loose=0.90, gs=2.65),
}
