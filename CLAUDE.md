# Project Notes

## Earth2Studio Framework

- **Repo**: https://github.com/NVIDIA/earth2studio
- **Docs**: https://nvidia.github.io/earth2studio/
- **Install guide**: https://nvidia.github.io/earth2studio/userguide/about/install.html
- **Installed version**: 0.12.1rc0 (check with `uv pip show earth2studio`)
- **Main branch** has newer features not in 0.12.1rc0 (e.g. `data_dtype` param, `_validate_time` in base class)

### Key source files (installed)

- Base class: `.venv/lib/python3.12/site-packages/earth2studio/data/planetary_computer.py`
- Lexicons: `.venv/lib/python3.12/site-packages/earth2studio/lexicon/planetary_computer.py`
- Data utils: `.venv/lib/python3.12/site-packages/earth2studio/data/utils.py`
- Lexicon metaclass: `.venv/lib/python3.12/site-packages/earth2studio/lexicon/base.py`

### Key source files (GitHub main, may differ from installed)

- https://raw.githubusercontent.com/NVIDIA/earth2studio/main/earth2studio/data/planetary_computer.py
- https://raw.githubusercontent.com/NVIDIA/earth2studio/main/earth2studio/lexicon/planetary_computer.py

### Planetary Computer data source architecture

`_PlanetaryComputerData` is the base class. Subclasses:
- `PlanetaryComputerOISST` — SST, 720×1440, NetCDF
- `PlanetaryComputerSentinel3AOD` — aerosol, 4040×324, NetCDF
- `PlanetaryComputerMODISFire` — fire, 1200×1200 sinusoidal, GeoTIFF
- `PlanetaryComputerECMWFOpenDataIFS` — IFS analysis, 721×1440, GRIB (main branch only)
- `PlanetaryComputerMetOffice` — our custom class in `metoffice_data.py`

Pipeline: `__call__` → `fetch` → `_fetch_data` → `_locate_item` + `_prepare_asset_plans` + `_downloaded_asset` + `extract_variable_numpy`

Key dataclasses: `AssetPlan` (download plan), `VariableSpec` (resolved variable request)

### Conventions to follow

- **Lexicons**: use `(dataset_key_str, modifier_callable)` tuples in VOCAB. Use named functions (not lambdas) for modifiers when building VOCAB in loops. See `ECMWFOpenDataIFSLexicon.build_vocab()` pattern with `nmod`/`zmod`.
- **Grid coords**: class-level `LAT_COORDS`/`LON_COORDS` (or `LATITUDE`/`LONGITUDE`), not module globals.
- **Time validation**: implement `_validate_time(times)`. On main branch, base class calls it from `fetch()`. On 0.12.1rc0, override `fetch()` to call it yourself.
- **Dataset key format**: varies by data source. ECMWF IFS uses `shortName::pressure::soilLayer`. Met Office uses `collection_type::asset_key::cf_variable[::pressure_hPa]`.

## Met Office Data Pipeline (Decomposed)

The Met Office data pipeline is split into two components:

### 1. Native Data Loader (`metoffice_native.py`)

**`PlanetaryComputerMetOfficeNative`** — pure DataSource, subclasses `_PlanetaryComputerData`.

- Returns raw Met Office fields on the **native** ~0.09° grid (1920×2560)
- Native coordinates: lat S→N [-90,90], lon [-180,180]
- 71 native variables: 6 surface + 5×13 pressure-level fields
- No derived variables, no regridding, no coordinate flipping
- Variable names: `air_temperature_500hPa`, `wind_speed_at_10m`, etc.
- Searches **two** STAC collections per timestamp (pressure + surface)
- Overrides: `_locate_item`, `_prepare_asset_plans`, `_fetch_data`, `extract_variable_numpy`, `fetch`, `cache`

#### STAC collections

- `met-office-global-deterministic-pressure` — pressure-level fields
- `met-office-global-deterministic-near-surface` — surface/screen-level fields
- Filter by `forecast:reference_datetime` and `forecast:horizon` (format: `PT0000H00M`)

#### Lexicon (`MetOfficeNativeLexicon`)

One-to-one mapping from native variable names to STAC assets. No derivation logic.
Dataset key format: `<collection_type>::<asset_key>::<cf_variable>[::pressure_hPa]`

### 2. Diagnostic Model (`metoffice_diagnostic.py`)

**`MetOfficeToAtlasDiagnostic`** — `torch.nn.Module` implementing `DiagnosticModel` protocol.

- 71 input variables (native Met Office) → 75 output variables (Atlas)
- All derivations on torch tensors, GPU-compatible
- Stateless, uses `@torch.inference_mode()`

#### Derivations

- **Wind components**: `u = -speed × sin(dir)`, `v = -speed × cos(dir)` (meteorological convention)
- **Specific humidity (q)**: from RH + T + P via Bolton (1980)
- **Geopotential (z)**: height × 9.80665
- **100 m wind**: falls back to 10 m (not available)
- **Surface pressure**: approximated by MSLP
- **TCWV**: filled with zeros (not available)

#### Regridding

Handled by the framework's `fetch_data(interp_to=model.input_coords())`, not by custom code.
Native ~0.09° → Atlas 0.25° (721×1440) via xarray interpolation.

### 3. Legacy Monolith (`metoffice_data.py`)

**`PlanetaryComputerMetOffice`** — original monolithic class (preserved for reference).
Combines LOAD + TRANSFORM in a single class with custom regridding.

### Composed Pipeline

```python
# 1. Fetch native Met Office data (framework regrids to Atlas grid)
x, coords = fetch_data(native_ds, time, diagnostic.input_coords()["variable"],
                       device=device, interp_to=atlas.input_coords())
# 2. Derive Atlas variables
x_atlas, coords_atlas = diagnostic(x, coords)
# 3. Feed into Atlas model
```
