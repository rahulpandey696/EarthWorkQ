# -*- coding: utf-8 -*-
"""
Balance level analysis for EarthWorkQ.

The question "at what formation level does the site balance?" is answered by
sweeping candidate levels and solving for the one where

    V_cut  =  shrinkage_factor  x  V_fill

Note the shrinkage factor.  Balancing raw cut against raw fill is the mistake
that leaves a site short of material: compacted fill occupies less space than
the same solids did in the ground, so more cubic metres have to be dug than
are placed.

The sweep is made effectively free by precomputing sorted cumulative sums of
the ground elevations weighted by cell coverage.  For a flat formation at
level t:

    V_fill(t) = A_cell * SUM_{g < t} (t - g) w = A_cell * ( t*W_below - S_below )
    V_cut(t)  = A_cell * SUM_{g > t} (g - t) w = A_cell * ( S_above - t*W_above )

Both are then O(log n) per level via a binary search, so a thousand candidate
levels cost nothing and the balance point can be found by bisection to machine
precision.
"""

from __future__ import annotations

import numpy as np


class BalanceCurve(object):
    """Cut, fill and net volume as a function of a vertical offset."""

    def __init__(self, offsets, cut, fill, shrinkage_factor=1.0,
                 reference_label="Formation level"):
        self.offsets = np.asarray(offsets, float)
        self.cut = np.asarray(cut, float)
        self.fill = np.asarray(fill, float)
        self.shrinkage_factor = float(shrinkage_factor)
        self.reference_label = reference_label

    @property
    def net(self):
        """Surplus (positive) or deficit (negative) after shrinkage."""
        return self.cut - self.shrinkage_factor * self.fill

    @property
    def net_raw(self):
        return self.cut - self.fill

    @property
    def total_movement(self):
        return self.cut + self.fill

    def balance_level(self):
        """Level at which the site balances, or None if it never does."""
        net = self.net
        sign = np.sign(net)
        change = np.where(np.diff(sign) != 0)[0]
        if change.size == 0:
            return None
        i = int(change[0])
        x0, x1 = self.offsets[i], self.offsets[i + 1]
        y0, y1 = net[i], net[i + 1]
        if y1 == y0:
            return float(x0)
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    def minimum_movement_level(self):
        """Level that minimises total earthmoving, ignoring the balance."""
        i = int(np.argmin(self.total_movement))
        return float(self.offsets[i])

    def at(self, offset):
        """Interpolated cut, fill and net at an arbitrary level."""
        return (float(np.interp(offset, self.offsets, self.cut)),
                float(np.interp(offset, self.offsets, self.fill)),
                float(np.interp(offset, self.offsets, self.net)))

    def to_rows(self):
        rows = [("level", "cut", "fill", "net_after_shrinkage", "total_movement")]
        for o, c, f, n, t in zip(self.offsets, self.cut, self.fill,
                                 self.net, self.total_movement):
            rows.append((o, c, f, n, t))
        return rows


class FlatLevelSweeper(object):
    """Fast exact sweep of a flat formation level over a weighted ground grid."""

    def __init__(self, ground, weight, cell_area):
        ground = np.asarray(ground, float)
        weight = np.asarray(weight, float)
        valid = np.isfinite(ground) & (weight > 0)
        g = ground[valid].ravel()
        w = weight[valid].ravel()
        order = np.argsort(g)
        self.g = g[order]
        self.w = w[order]
        self.cell_area = float(cell_area)
        self.cum_w = np.concatenate(([0.0], np.cumsum(self.w)))
        self.cum_gw = np.concatenate(([0.0], np.cumsum(self.g * self.w)))
        self.total_w = self.cum_w[-1]
        self.total_gw = self.cum_gw[-1]

    @property
    def has_data(self):
        return self.g.size > 0

    def z_range(self):
        return float(self.g.min()), float(self.g.max())

    def cut_fill(self, level):
        """Exact cut and fill volumes for a flat formation at ``level``."""
        i = int(np.searchsorted(self.g, level, side="right"))
        w_below = self.cum_w[i]
        gw_below = self.cum_gw[i]
        w_above = self.total_w - w_below
        gw_above = self.total_gw - gw_below
        fill = (level * w_below - gw_below) * self.cell_area
        cut = (gw_above - level * w_above) * self.cell_area
        return max(cut, 0.0), max(fill, 0.0)

    def sweep(self, levels, shrinkage_factor=1.0):
        cuts, fills = [], []
        for lv in levels:
            c, f = self.cut_fill(lv)
            cuts.append(c)
            fills.append(f)
        return BalanceCurve(levels, cuts, fills, shrinkage_factor)

    def solve_balance(self, shrinkage_factor=1.0, tol=1e-6, max_iter=200):
        """Bisection for cut = SF * fill. The function is monotone decreasing."""
        lo, hi = self.z_range()
        if lo == hi:
            return lo

        def net(t):
            c, f = self.cut_fill(t)
            return c - shrinkage_factor * f

        n_lo, n_hi = net(lo), net(hi)
        if n_lo <= 0:
            return lo           # all fill even at the lowest ground point
        if n_hi >= 0:
            return hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            n_mid = net(mid)
            if abs(n_mid) < tol or (hi - lo) < 1e-9:
                return mid
            if n_mid > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def sweep_levels(ground, weight, cell_area, n_steps=200,
                 shrinkage_factor=1.0, pad_fraction=0.02):
    """Build a balance curve across the full range of ground elevations."""
    sweeper = FlatLevelSweeper(ground, weight, cell_area)
    if not sweeper.has_data:
        return None, None
    lo, hi = sweeper.z_range()
    pad = (hi - lo) * pad_fraction
    levels = np.linspace(lo - pad, hi + pad, int(n_steps))
    curve = sweeper.sweep(levels, shrinkage_factor)
    return curve, sweeper


def sweep_offsets(ground, design_grid, weight, cell_area, offsets,
                  shrinkage_factor=1.0):
    """Sweep a bodily vertical shift of an arbitrary design surface.

    Used when the design is a sloped plane or a design DEM: the shape is fixed
    and only its elevation is varied.  Equivalent to sweeping a flat level on
    the residual surface (ground - design), so the same fast sweeper applies.
    """
    residual = np.asarray(ground, float) - np.asarray(design_grid, float)
    sweeper = FlatLevelSweeper(residual, weight, cell_area)
    if not sweeper.has_data:
        return None, None
    curve = sweeper.sweep(np.asarray(offsets, float), shrinkage_factor)
    curve.reference_label = "Vertical offset applied to design surface"
    return curve, sweeper


def auto_offset_range(ground, design_grid, weight, n_steps=200,
                      pad_fraction=0.05):
    residual = np.asarray(ground, float) - np.asarray(design_grid, float)
    valid = np.isfinite(residual) & (np.asarray(weight, float) > 0)
    if not valid.any():
        return np.linspace(-1.0, 1.0, n_steps)
    r = residual[valid]
    lo, hi = float(r.min()), float(r.max())
    pad = max((hi - lo) * pad_fraction, 0.01)
    return np.linspace(lo - pad, hi + pad, int(n_steps))


def haul_summary(result, soil):
    """Material movement summary for the site, in the units the site uses."""
    if soil is None:
        return None
    required_cut = soil.cut_for_fill(result.fill)
    surplus = result.cut - required_cut
    rows = [
        ("Cut (in-situ)", result.cut, "m3"),
        ("Fill (compacted, in place)", result.fill, "m3"),
        ("Cut required to make that fill", required_cut, "m3"),
        ("Surplus for disposal" if surplus >= 0 else "Deficit to import",
         abs(surplus), "m3"),
    ]
    loose = soil.loose_volume_from_cut(abs(surplus))
    if loose is not None:
        rows.append(("Surplus / deficit as loose volume", loose, "m3 loose"))
    mass = soil.haul_mass_from_cut(abs(surplus))
    if mass is not None:
        rows.append(("Surplus / deficit haul mass", mass, "t"))
    water = soil.conditioning_water(result.fill)
    if water is not None:
        rows.append(("Conditioning water to add" if water >= 0
                     else "Moisture to remove before placing",
                     abs(water), "m3"))
    return rows
