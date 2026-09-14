# EarthWorkQ

Earthwork cut and fill quantities for QGIS, computed by coverage-weighted grid
integration, converted between cut and fill on a void-ratio basis, and issued
with a calculation report that records every input, assumption and check.

- **Author** — Rahul Pandey (<rahulpandey696@gmail.com>)
- **Repository** — https://github.com/rahulpandey696/EarthWorkQ
- **Issues** — https://github.com/rahulpandey696/EarthWorkQ/issues
- **Licence** — GNU General Public License v3.0 or later
- **Requires** — QGIS 3.22 or later, including QGIS 4.x (Qt5 and Qt6)

## Scope

Site platform earthworks: the volume of material to be excavated and placed
between a surveyed existing ground surface and a design formation, over one or
more polygons, together with the formation level at which the site balances.

Chainage-based cross-sections and mass-haul diagrams for linear works are out of
scope.

## Inputs

### Existing ground

| Source | Handling |
|---|---|
| CSV / XLSX table | Easting, northing, elevation. Headings matched against common aliases; CRS must be stated explicitly since the file carries none. Non-numeric rows discarded and counted; coincident XY with differing Z rejected as a transcription error. |
| Point layer | Elevation from a numeric attribute or from geometry Z. Local outliers flagged by a modified Z-score of the residual against the k nearest neighbours. |
| Contour polylines | Densified before fitting. Topology validated for crossing lines at differing elevations. Interval detected and the implied vertical accuracy reported. |
| DEM raster | Resampled onto the working grid by bilinear warp. No interpolation applied. |

All inputs must be in a projected CRS with map units in metres or feet. A
geographic CRS is rejected outright — areas and volumes cannot be derived from
degrees.

### Design surface

- Flat formation level
- Sloped plane: `z = z₀ + gₓ(x − x₀) + g_y(y − y₀)`, grades entered as percentages
- Design DEM, with an optional bodily vertical offset

The formation level may also be solved for, so that the site balances.

### Material

Specified by void ratios, by dry densities, or by an in-situ density together
with a compaction requirement as a percentage of Proctor MDD. Optional: loose
void ratio, natural and target water content, topsoil strip depth.

## Method

### Volume integration

Everything is reduced to aligned arrays on a common grid: existing ground,
design surface, and a per-cell coverage weight. With `d = z_design − z_ground`:

```
V_fill = Σ max(d, 0) · w · A_cell
V_cut  = Σ max(−d, 0) · w · A_cell
```

The coverage weight `w ∈ [0,1]` is obtained by rasterising the area of interest
at 8× resolution and block-averaging. Including or excluding boundary cells
whole on a centre-in-polygon test introduces an error proportional to the
perimeter of the site, which for a corridor or a channel is not a rounding
difference.

Default cell size is half the mean nearest-neighbour spacing of the survey. A
finer grid adds no information the survey does not contain, and the report says
so if one is forced.

### No extrapolation

Cells outside the convex hull of the survey, or beyond the outermost contour,
are returned as nodata and excluded. The uncovered area is reported; above 5% of
the area of interest it is raised as an error rather than quietly averaged over.

### Cut/fill conversion

Volume of solids is conserved between cut and fill; water is not, being added or
driven off during conditioning and compaction. With `V_solids = V_total/(1+e)`:

```
shrinkage factor = V_cut / V_fill = (1 + e_in-situ) / (1 + e_compacted)
```

Typically 1.10–1.30 for soils, and below 1.0 for rockfill, which swells. The
balance condition is therefore `V_cut = SF · V_fill`, not `V_cut = V_fill` —
balancing raw volumes is what leaves a site short of material.

Water content enters only three calculations: conditioning water
`M_s(w_target − w_natural)`, haul mass `M_s(1 + w)`, and the validity check
`S = w·G_s/e ≤ 1`, which rejects parameter sets describing a physically
impossible soil.

### Interpolation

| Engine | Exact | Notes |
|---|---|---|
| Clough-Tocher TIN | yes | Cubic patch per triangle, C¹ across edges. Default for scattered points. |
| Linear Delaunay TIN | yes | Planar facets. The method underlying traditional spot-level volumes. |
| Thin-plate RBF | no | Tension and smoothing; lets noisy GNSS data be filtered rather than honoured. |
| IDW | yes | Cannot exceed the range of its neighbours, so ridges flatten and hollows lift. Not recommended for terrain. |
| Nearest neighbour | yes | Coverage diagnostic only. |
| GRASS `v.surf.rst` | no | Segmented regularised spline with tension. |
| GRASS `r.surf.contour` | yes | Default for contour input. |
| QGIS TIN | yes | Fallback when SciPy is absent. |

Ten-fold cross-validation runs across every applicable engine and reports RMSE,
MAE, bias and maximum error, turning the choice of interpolator into a
measurement. Contour input defaults to `r.surf.contour` because it
distance-weights between bracketing contours instead of triangulating vertices,
avoiding the flat-triangle artefacts that give level hilltops and dead-flat
valley floors.

### Balance analysis

Sorting the coverage-weighted ground elevations and precomputing cumulative sums
reduces cut and fill at any level `t` to a binary search:

```
V_fill(t) = A_cell · (t·W_below − S_below)
V_cut(t)  = A_cell · (S_above − t·W_above)
```

A full sweep therefore costs nothing, and the balance level is found by
bisection on a monotone function. The level of minimum total earthmoving is
reported alongside it; the two are rarely the same, and choosing between them is
a cost decision — hauling material off site against double-handling it on site.

For a sloped plane or a design DEM the same machinery applies to the residual
surface `z_ground − z_design`, sweeping a bodily vertical offset.

### Verification

- **Grid convergence** — quantities recomputed at cell, cell/2 and cell/4. Still
  moving means the cell size is too coarse to issue, and the report says so.
- **Exact TIN prismatic integration** — where the design is planar, `f = z_design
  − z_ground` is linear over each Delaunay triangle. Triangles straddling the
  zero line are split at the intersection; the sub-triangle containing the odd
  vertex has two vertices at `f = 0`, so its volume is `A · f_odd/3`. Wholly
  independent of the grid result.
- **Volume uncertainty** — interpolation RMSE propagated as `σ_V ≈ RMSE·A/√n`.
  Deliberately an order-of-magnitude statement, not a geostatistical variance.

## Outputs

- Signed cut/fill depth raster, diverging ramp centred on zero
- Existing and design surface rasters
- Cut / balanced / fill zone polygons with areas
- Depth contours of the difference surface
- Per-polygon quantities as CSV and written back as layer attributes
- Balance curve chart and CSV; area-elevation (hypsometric) curve
- HTML calculation report with charts embedded

## Processing algorithms

Registered under the **EarthWorkQ** provider, so the same code runs in the model
builder, in batch, and from PyQGIS:

- *Earthwork quantities (cut and fill)*
- *Balance level analysis*
- *Build a surface from survey data*

## Dependencies

| Package | Needed for |
|---|---|
| NumPy | everything (ships with QGIS) |
| GDAL | rasterisation, warping, raster I/O (ships with QGIS) |
| SciPy | TIN, spline and IDW engines, cross-validation, TIN cross-check |
| matplotlib | balance and hypsometric charts |
| openpyxl | `.xlsx` input |
| GRASS provider | `r.surf.contour`, `v.surf.rst` |

Missing optional packages degrade gracefully — affected engines and outputs are
hidden or skipped with a note rather than failing the run. The *Method and
checks* tab lists what was found.

## Tests

```
python tests/test_core.py
```

Twenty-two tests, no QGIS installation required. Volumes are asserted against
surfaces with closed-form solutions — a tilted plane, a cone, a paraboloid — and
against convergence as the grid is refined, rather than against golden values.
The soil tests assert conservation of solids directly.

## Limitations

- Exact TIN cross-check applies only to planar design surfaces.
- Volume uncertainty covers interpolation error alone. Survey error, material
  variability and construction tolerance are additional.
- Contour input carries a vertical accuracy of roughly half the contour
  interval, which integrated over a site normally dominates every other error in
  the result — including anything the choice of interpolation engine
  contributes.
- Quantities are only as good as the survey behind them. The report exists so
  that a reviewer can judge that for themselves.
