# TCWV Derivation from Met Office Pressure-Level Data

## Overview

Total Column Water Vapor (TCWV, kg m⁻²) is required by the Atlas model but is
not published directly in any of the four Met Office Global Deterministic
collections on Planetary Computer (pressure, near-surface, whole-atmosphere,
height). Instead, we derive it by vertically integrating specific humidity over
the 13 Atlas pressure levels.

## Method

TCWV is the mass of water vapor in a column per unit area:

```
TCWV = (1/g) ∫₀ᵖˢ q dp
```

where `q` is specific humidity (kg kg⁻¹), `p` is pressure (Pa), and
`g = 9.80665 m s⁻²`.

We approximate this with the **trapezoidal rule** over the 13 pressure levels
already fetched for Atlas (50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
850, 925, 1000 hPa):

```
TCWV ≈ (1/g) Σᵢ ½ (qᵢ + qᵢ₊₁)(pᵢ₊₁ - pᵢ)
```

Specific humidity at each level is already computed in the diagnostic from
relative humidity and temperature via Bolton (1980) saturation vapor pressure
(see `MetOfficeAtlasDiagnostic._specific_humidity`).

Implementation: `MetOfficeAtlasDiagnostic._integrate_tcwv` in
`src/metoffice_diagnostic.py`.

## Available pressure levels

The Met Office publishes **37 pressure levels** for temperature/wind and **33
for relative humidity** (RH stops at 10 hPa; the top 4 levels at 5, 2, 1, 0.4
hPa have no RH). We use only the 13 that Atlas needs for its other variables
(u, v, t, z, q), avoiding extra data fetches.

All levels come from the same NetCDF asset per variable, so adding intermediate
levels would not require additional STAC lookups or HTTP downloads — only
additional slices from the same file. This is a possible future optimization if
higher accuracy is needed.

## Error analysis

We estimated the integration error by comparing 13-level vs 33-level trapezoidal
integration using realistic specific humidity profiles representative of three
climate regimes:

| Profile      | TCWV (33-level) | TCWV (13-level) | Error  |
|-------------|-----------------|-----------------|--------|
| Tropical     | 74.93 kg/m²     | 74.70 kg/m²     | -0.3%  |
| Mid-latitude | 31.46 kg/m²     | 30.35 kg/m²     | -3.5%  |
| Polar/dry    | 6.66 kg/m²      | 6.25 kg/m²      | -6.3%  |

The 13-level integration consistently **underestimates** TCWV slightly because
the trapezoidal rule assumes linear `q` between levels, while real `q` profiles
are slightly convex (higher curvature in drier columns).

### Per-layer breakdown (tropical profile)

| Layer (hPa) | 33-level (kg/m²) | 13-level (kg/m²) | Error  |
|-------------|-----------------|-----------------|--------|
| 700–850     | 20.90           | 20.65           | -1.2%  |
| 850–925     | 12.43           | 12.43           | 0.0%  |
| 925–1000    | 13.57           | 13.57           | 0.0%  |

The 700–850 hPa gap (150 hPa, the largest in the moist lower troposphere) is the
primary error source, but contributes only ~1% error even in this layer. Upper
tropospheric layers contribute negligible error because `q` is small there.

## Assessment

The 13-level trapezoidal integration is accurate to within ~0–6% across climate
regimes, with a slight low bias. This is well within acceptable bounds for Atlas
model input, where TCWV is one of 75 prognostic variables and the model is
tolerant of input approximations (the previous implementation filled TCWV with
zeros).

If higher accuracy is needed in the future, the most impactful improvement would
be adding the 750 and 800 hPa levels to reduce the 700–850 hPa gap, at no
additional download cost.
