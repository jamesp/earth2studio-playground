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

## Met Office Data Source (`metoffice_data.py`)

### Design

- Subclasses `_PlanetaryComputerData`
- Searches **two** STAC collections per timestamp (pressure + surface)
- Multiple assets per timestamp (temperature, wind speed, wind direction, RH, geopotential height, etc.)
- Overrides: `_locate_item`, `_prepare_asset_plans`, `_fetch_data`, `extract_variable_numpy`, `fetch`, `cache`

### STAC collections

- `met-office-global-deterministic-pressure` — pressure-level fields
- `met-office-global-deterministic-near-surface` — surface/screen-level fields
- Filter by `forecast:reference_datetime` and `forecast:horizon` (format: `PT0000H00M`)

### Derived variables (multi-asset)

- **u/v wind**: decomposed from wind speed + direction assets (meteorological convention: direction is "from", clockwise from north)
- **Specific humidity (q)**: derived from relative humidity + temperature + pressure via Bolton (1980) saturation vapour pressure
- **Geopotential (z)**: height × 9.80665

### Known approximations

- `sp` (surface pressure) → uses MSLP (no orographic reduction)
- `u100m`/`v100m` → falls back to 10m winds
- `tcwv` → filled with zeros (not available)

### Grid

- Native: ~0.09° (1920×2560), lat S→N, lon [-180, 180]
- Output: 0.25° (721×1440), lat N→S, lon [0, 360)
- Regridding: bilinear via `scipy.interpolate.RegularGridInterpolator`

### Key gotcha: `_prepare_asset_plans` deduplication

Derived variables need multiple assets (e.g. u500 needs both speed and direction). Each `VariableSpec` must be assigned to exactly **one** plan to avoid duplicate extraction. The `extract_variable_numpy` method resolves sibling assets independently via `_read_field` + `_current_items`.

Constant-fill variables (e.g. `tcwv`) have no backing assets and are handled separately in `_fetch_data`.
