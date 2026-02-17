# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""Earth2Studio data source for Met Office global deterministic forecast data
hosted on Microsoft Planetary Computer.

This module provides a :class:`PlanetaryComputerMetOffice` data source that
subclasses :class:`earth2studio.data.planetary_computer._PlanetaryComputerData`
and fetches pressure-level and near-surface fields from two Met Office STAC
collections.

The data source handles:

- Wind speed + direction → u/v component decomposition
- Relative humidity → specific humidity conversion
- Geopotential height (m) → geopotential (m² s⁻²) conversion
- Regridding from the native ~0.09° grid to the 0.25° 721×1440 Atlas grid
- Longitude convention shift from [-180, 180] to [0, 360)

Usage::

    from metoffice_data import PlanetaryComputerMetOffice
    from datetime import datetime

    ds = PlanetaryComputerMetOffice()
    da = ds(datetime(2026, 2, 17), ["t2m", "u500", "z500", "msl"])
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.data.planetary_computer import (
    AssetPlan,
    VariableSpec,
    _PlanetaryComputerData,
)
from earth2studio.data.utils import datasource_cache_root
from earth2studio.lexicon.base import LexiconType

try:
    import planetary_computer
    from pystac_client import Client
except ImportError:
    planetary_computer = None
    Client = None

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Standard gravity (m s⁻²), used for geopotential height → geopotential.
G = np.float32(9.80665)

#: Ratio of molecular weight of water vapour to dry air.
EPSILON = np.float32(0.622)

#: Pressure levels served by the Atlas model (hPa).
ATLAS_PRESSURE_LEVELS_HPA: list[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]

#: Mapping from Atlas hPa to Met Office Pa coordinate values.
_HPA_TO_PA: dict[int, float] = {hpa: float(hpa * 100) for hpa in ATLAS_PRESSURE_LEVELS_HPA}

# ---------------------------------------------------------------------------
# Met Office field names (CF standard names in the NetCDF assets)
# ---------------------------------------------------------------------------

# Pressure-level asset keys and CF variable names
_PRESSURE_ASSETS: dict[str, str] = {
    "geopotential_height_on_pressure_levels": "geopotential_height",
    "temperature_on_pressure_levels": "air_temperature",
    "wind_speed_on_pressure_levels": "wind_speed",
    "wind_direction_on_pressure_levels": "wind_from_direction",
    "relative_humidity_on_pressure_levels": "relative_humidity",
}

# Surface asset keys and CF variable names
_SURFACE_ASSETS: dict[str, str] = {
    "temperature_at_screen_level": "air_temperature",
    "pressure_at_mean_sea_level": "air_pressure_at_sea_level",
    "wind_speed_at_10m": "wind_speed",
    "wind_direction_at_10m": "wind_from_direction",
    "temperature_at_surface": "surface_temperature",
    "precipitation_rate": "lwe_precipitation_rate",
}


# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

def _saturation_vapor_pressure(t_k: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (Pa) from temperature (K) via Bolton (1980)."""
    t_c = t_k - np.float32(273.15)
    return np.float32(611.2) * np.exp(
        np.float32(17.67) * t_c / (t_c + np.float32(243.5))
    )


def _specific_humidity(
    rh_frac: np.ndarray, t_k: np.ndarray, p_pa: np.ndarray,
) -> np.ndarray:
    """Specific humidity (kg kg⁻¹) from fractional RH, T (K), and P (Pa)."""
    e_s = _saturation_vapor_pressure(t_k)
    e = rh_frac * e_s
    q = EPSILON * e / (p_pa - (np.float32(1.0) - EPSILON) * e)
    return np.clip(q, 0.0, None).astype(np.float32)


def _wind_components(
    speed: np.ndarray, direction_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological wind speed + direction (degrees, 'from') → (u, v).

    Convention: direction is where the wind blows FROM, clockwise from north.
    ``u = -speed × sin(dir)``, ``v = -speed × cos(dir)``.
    """
    direction_rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)
    return u.astype(np.float32), v.astype(np.float32)


# ---------------------------------------------------------------------------
# Modifier helpers (used by the lexicon)
# ---------------------------------------------------------------------------

Modifier = Callable[[Any], Any]


def _nmod(x: Any) -> Any:
    """Identity modifier — return the input unchanged."""
    return x


def _zmod(x: np.ndarray) -> np.ndarray:
    """Geopotential height (m) → geopotential (m² s⁻²)."""
    return x * G


# ---------------------------------------------------------------------------
# Lexicon
#
# The ``dataset_key`` string uses the format:
#   <collection_type>::<asset_key>::<cf_variable>[::pressure_hPa]
#
# where <collection_type> is "pressure" or "surface".
# Derived variables (u/v wind components, specific humidity) are *not*
# encoded in the lexicon.  Instead, the data source recognises them by
# their Earth2Studio name prefix (u*, v*, q*) and applies the
# appropriate multi-asset derivation logic at extraction time.
# ---------------------------------------------------------------------------


def _build_vocab() -> dict[str, tuple[str, Modifier]]:
    """Build the full E2S-variable → Met Office mapping."""
    vocab: dict[str, tuple[str, Modifier]] = {}

    # -- Near-surface direct variables --
    vocab["t2m"] = ("surface::temperature_at_screen_level::air_temperature", _nmod)
    vocab["msl"] = (
        "surface::pressure_at_mean_sea_level::air_pressure_at_sea_level", _nmod,
    )
    vocab["sst"] = ("surface::temperature_at_surface::surface_temperature", _nmod)
    vocab["tp"] = ("surface::precipitation_rate::lwe_precipitation_rate", _nmod)

    # sp: surface pressure — Met Office provides MSLP only.  We use MSLP as
    # an approximation.  A proper implementation would require orographic
    # height and temperature to reduce MSLP to surface pressure.
    vocab["sp"] = (
        "surface::pressure_at_mean_sea_level::air_pressure_at_sea_level", _nmod,
    )

    # 10 m wind: derived from speed + direction at extraction time
    vocab["u10m"] = ("surface::wind_speed_at_10m::wind_speed", _nmod)
    vocab["v10m"] = ("surface::wind_speed_at_10m::wind_speed", _nmod)

    # 100 m wind: not available; fall back to 10 m
    vocab["u100m"] = ("surface::wind_speed_at_10m::wind_speed", _nmod)
    vocab["v100m"] = ("surface::wind_speed_at_10m::wind_speed", _nmod)

    # tcwv: not available — filled with zeros at extraction time
    vocab["tcwv"] = ("constant::0.0", _nmod)

    # -- Pressure-level variables --
    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        vocab[f"z{hpa}"] = (
            f"pressure::geopotential_height_on_pressure_levels"
            f"::geopotential_height::{hpa}",
            _zmod,
        )
        vocab[f"t{hpa}"] = (
            f"pressure::temperature_on_pressure_levels"
            f"::air_temperature::{hpa}",
            _nmod,
        )
        # u, v, q are resolved via multi-asset derivation; the dataset_key
        # points to a representative asset for the purpose of the STAC plan,
        # but extraction is handled by specialised helpers.
        vocab[f"u{hpa}"] = (
            f"pressure::wind_speed_on_pressure_levels::wind_speed::{hpa}",
            _nmod,
        )
        vocab[f"v{hpa}"] = (
            f"pressure::wind_speed_on_pressure_levels::wind_speed::{hpa}",
            _nmod,
        )
        vocab[f"q{hpa}"] = (
            f"pressure::temperature_on_pressure_levels::air_temperature::{hpa}",
            _nmod,
        )

    return vocab


class MetOfficeLexicon(metaclass=LexiconType):
    """Lexicon mapping Earth2Studio variable names to Met Office data fields."""

    VOCAB = _build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        return cls.VOCAB[val]


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------


class PlanetaryComputerMetOffice(_PlanetaryComputerData):
    """Earth2Studio DataSource for Met Office Global 10 km Deterministic
    forecast data from Microsoft Planetary Computer.

    Fetches forecast fields from the Met Office global model and returns
    them on the 0.25° 721×1440 grid used by NVIDIA Atlas.

    The data source searches **two** STAC collections per timestamp
    (pressure-level and near-surface), downloads the required NetCDF
    assets, applies physical transformations, and regrids to the output
    grid via bilinear interpolation.

    Parameters
    ----------
    forecast_hour : int, optional
        Forecast lead time in hours, by default 0 (T+0 / analysis-like).
    cache : bool, optional
        Cache downloaded files locally, by default True.
    verbose : bool, optional
        Print progress information, by default True.
    max_workers : int, optional
        Maximum concurrent downloads, by default 8.
    request_timeout : int, optional
        HTTP request timeout in seconds, by default 120.
    max_retries : int, optional
        Maximum HTTP retry attempts, by default 4.

    Note
    ----
    Known limitations:

    - ``tcwv`` (total column water vapour) is not available and is filled
      with zeros.
    - ``u100m`` / ``v100m`` (100 m wind) fall back to 10 m winds.
    - ``sp`` (surface pressure) uses MSLP as an approximation.

    Example
    -------
    >>> from metoffice_data import PlanetaryComputerMetOffice
    >>> from datetime import datetime
    >>> ds = PlanetaryComputerMetOffice()
    >>> da = ds(datetime(2026, 2, 17), ["t2m", "u500", "z500"])
    """

    # -- STAC collections --
    PRESSURE_COLLECTION = "met-office-global-deterministic-pressure"
    SURFACE_COLLECTION = "met-office-global-deterministic-near-surface"

    # -- Output grid (Atlas 0.25°) --
    LAT_COORDS = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    LON_COORDS = np.linspace(0.0, 360.0, 1440, dtype=np.float32, endpoint=False)

    # Met Office runs are produced every 6 hours.
    _VALID_RUN_HOURS = {0, 6, 12, 18}

    def __init__(
        self,
        forecast_hour: int = 0,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 8,
        request_timeout: int = 120,
        max_retries: int = _PlanetaryComputerData.DEFAULT_RETRIES,
    ) -> None:
        # The base class expects a single collection_id; we pass the
        # pressure collection as the primary but override _locate_item
        # and _prepare_asset_plans to query both collections.
        super().__init__(
            collection_id=self.PRESSURE_COLLECTION,
            lexicon=MetOfficeLexicon,
            asset_key="temperature_on_pressure_levels",  # nominal; overridden
            search_tolerance=timedelta(hours=0),
            spatial_dims={
                "lat": self.LAT_COORDS,
                "lon": self.LON_COORDS,
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        self._forecast_hour = forecast_hour
        # Instance-level dataset cache for the current timestamp, avoids
        # re-opening large NetCDF files when extracting many variables.
        self._ds_cache: dict[str, xr.Dataset] = {}
        # Regrid interpolator state (lazily initialised)
        self._regrid_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Time validation & fetch override
    # ------------------------------------------------------------------

    async def fetch(
        self,
        time: datetime | list[datetime],
        variable: str | list[str],
    ) -> xr.DataArray:
        """Validate times then delegate to the base-class fetch pipeline."""
        from earth2studio.data.utils import prep_data_inputs

        times, _ = prep_data_inputs(time, variable)
        self._validate_time(times)
        return await super().fetch(time, variable)

    @staticmethod
    def _validate_time(times: list[datetime]) -> None:
        """Verify requested times are valid Met Office reference times.

        Met Office global deterministic forecasts are issued every 6 hours
        (00, 06, 12, 18 UTC).

        Parameters
        ----------
        times : list[datetime]
            Reference times to validate.

        Raises
        ------
        ValueError
            If any time is not on a 6-hour boundary.
        """
        for t in times:
            if t.hour not in PlanetaryComputerMetOffice._VALID_RUN_HOURS or t.minute or t.second:
                raise ValueError(
                    f"Met Office reference time {t} must be on a 6-hour boundary "
                    f"(00, 06, 12, or 18 UTC)"
                )

    # ------------------------------------------------------------------
    # STAC search (two collections)
    # ------------------------------------------------------------------

    def _locate_item(self, when: datetime) -> dict[str, Any]:
        """Locate STAC items for *both* collections at the given reference time.

        Overrides the base-class single-collection search.

        Returns
        -------
        dict[str, Any]
            Mapping ``{collection_id: pystac.Item}``.
        """
        if self._client is None:
            self._client = Client.open(self.STAC_API_URL)

        ref_time_str = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        horizon_str = f"PT{self._forecast_hour:04d}H00M"

        items: dict[str, Any] = {}
        for collection in (self.PRESSURE_COLLECTION, self.SURFACE_COLLECTION):
            search = self._client.search(
                collections=[collection],
                limit=1,
                filter={
                    "op": "and",
                    "args": [
                        {
                            "op": "=",
                            "args": [
                                {"property": "forecast:reference_datetime"},
                                ref_time_str,
                            ],
                        },
                        {
                            "op": "=",
                            "args": [
                                {"property": "forecast:horizon"},
                                horizon_str,
                            ],
                        },
                    ],
                },
            )
            try:
                items[collection] = next(search.items())
            except StopIteration:
                raise FileNotFoundError(
                    f"No Met Office item for ref={ref_time_str} "
                    f"horizon={horizon_str} in {collection}"
                )
        return items

    # ------------------------------------------------------------------
    # Asset planning (multiple assets per timestamp)
    # ------------------------------------------------------------------

    def _prepare_asset_plans(
        self,
        items: Any,  # dict[str, pystac.Item] from _locate_item
        variables: Sequence[VariableSpec],
    ) -> list[AssetPlan]:
        """Build download plans for all assets required by *variables*.

        Each unique (collection, asset_key) pair produces one
        :class:`AssetPlan`.  Each :class:`VariableSpec` is assigned to
        exactly one plan (the first asset it requires) to avoid duplicate
        extraction calls; :meth:`extract_variable_numpy` independently
        resolves all sibling assets via :meth:`_read_field`.

        Overrides the base-class method to handle the Met Office's
        per-variable-group asset layout across two collections.
        """
        # Collect all (collection_id, asset_key) pairs needed for downloads,
        # and assign each spec to exactly one plan.
        needed_assets: dict[tuple[str, str], list[VariableSpec]] = {}
        assigned: set[int] = set()  # spec indices already assigned to a plan

        for spec in variables:
            asset_keys = self._spec_to_asset_keys(spec)
            for col, asset_key in asset_keys:
                needed_assets.setdefault((col, asset_key), [])
            # Assign spec to its first asset's plan (avoid double extraction)
            if asset_keys and spec.index not in assigned:
                first_key = asset_keys[0]
                needed_assets[first_key].append(spec)
                assigned.add(spec.index)

        plans: list[AssetPlan] = []
        for (collection, asset_key), specs in needed_assets.items():
            item = items[collection]
            asset = item.assets[asset_key]
            signed = planetary_computer.sign(asset)  # type: ignore[union-attr]
            local_path = self._local_asset_path(asset.href)
            plans.append(
                AssetPlan(
                    unsigned_href=asset.href,
                    signed_href=signed.href,
                    media_type=asset.media_type,
                    local_path=local_path,
                    variables=specs,
                )
            )
        return plans

    @staticmethod
    def _spec_to_asset_keys(
        spec: VariableSpec,
    ) -> list[tuple[str, str]]:
        """Determine the (collection, asset_key) pairs a variable needs."""
        ds_key = spec.dataset_key
        var_id = spec.variable_id

        if ds_key.startswith("constant::"):
            return []

        parts = ds_key.split("::")
        collection_type = parts[0]  # "pressure" or "surface"
        col = (
            PlanetaryComputerMetOffice.PRESSURE_COLLECTION
            if collection_type == "pressure"
            else PlanetaryComputerMetOffice.SURFACE_COLLECTION
        )

        keys: list[tuple[str, str]] = []

        # Wind variables need both speed + direction assets
        if var_id.startswith(("u", "v")):
            if collection_type == "pressure":
                keys.append((col, "wind_speed_on_pressure_levels"))
                keys.append((col, "wind_direction_on_pressure_levels"))
            else:
                keys.append((col, "wind_speed_at_10m"))
                keys.append((col, "wind_direction_at_10m"))
            return keys

        # Specific humidity needs temperature + relative humidity
        if var_id.startswith("q") and collection_type == "pressure":
            keys.append((col, "temperature_on_pressure_levels"))
            keys.append((col, "relative_humidity_on_pressure_levels"))
            return keys

        # Direct single-asset variable
        asset_key = parts[1]
        keys.append((col, asset_key))
        return keys

    # ------------------------------------------------------------------
    # Variable extraction
    # ------------------------------------------------------------------

    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        _target_time: datetime,
    ) -> np.ndarray:
        """Extract a single variable from cached NetCDF assets.

        Handles direct reads, wind decomposition, and humidity derivation,
        then regrids the native Met Office field to the Atlas output grid.

        Parameters
        ----------
        plan : AssetPlan
            The asset plan that contains *spec*.  For derived variables
            (wind components, specific humidity) additional sibling assets
            are resolved from the same cache directory.
        spec : VariableSpec
            Variable specification from the lexicon.
        _target_time : datetime
            Request timestamp (unused here).

        Returns
        -------
        np.ndarray
            Float32 array shaped ``(721, 1440)``.
        """
        var_id = spec.variable_id
        ds_key = spec.dataset_key

        # -- constant fill --
        if ds_key.startswith("constant::"):
            value = float(ds_key.split("::")[1])
            return np.full(self._spatial_shape, value, dtype=np.float32)

        parts = ds_key.split("::")
        collection_type = parts[0]
        pressure_hpa = int(parts[3]) if len(parts) > 3 else None

        col = (
            self.PRESSURE_COLLECTION
            if collection_type == "pressure"
            else self.SURFACE_COLLECTION
        )

        # -- 10 m wind components --
        if var_id in ("u10m", "v10m", "u100m", "v100m"):
            if var_id in ("u100m", "v100m"):
                logger.warning(
                    f"{var_id}: 100 m wind not available from Met Office; "
                    "falling back to 10 m wind"
                )
            speed = self._read_field(col, "wind_speed_at_10m", "wind_speed", None)
            dirn = self._read_field(col, "wind_direction_at_10m", "wind_from_direction", None)
            u, v = _wind_components(speed, dirn)
            raw = u if var_id.startswith("u") else v
            return self._regrid(raw)

        # -- pressure-level wind components --
        if var_id[0] in ("u", "v") and pressure_hpa is not None:
            speed = self._read_field(
                col, "wind_speed_on_pressure_levels", "wind_speed", pressure_hpa,
            )
            dirn = self._read_field(
                col, "wind_direction_on_pressure_levels", "wind_from_direction",
                pressure_hpa,
            )
            u, v = _wind_components(speed, dirn)
            raw = u if var_id.startswith("u") else v
            return self._regrid(raw)

        # -- specific humidity --
        if var_id.startswith("q") and pressure_hpa is not None:
            t_k = self._read_field(
                col, "temperature_on_pressure_levels", "air_temperature",
                pressure_hpa,
            )
            rh = self._read_field(
                col, "relative_humidity_on_pressure_levels", "relative_humidity",
                pressure_hpa,
            )
            q = _specific_humidity(rh, t_k, np.float32(pressure_hpa * 100))
            return self._regrid(q)

        # -- surface pressure approximation --
        if var_id == "sp":
            logger.warning(
                "sp: surface pressure approximated by MSLP; a proper reduction "
                "would require orographic height and temperature"
            )

        # -- direct single-field read --
        asset_key = parts[1]
        cf_var = parts[2]
        raw = self._read_field(col, asset_key, cf_var, pressure_hpa)
        raw = spec.modifier(raw)
        return self._regrid(raw)

    # ------------------------------------------------------------------
    # NetCDF I/O with instance-level caching
    # ------------------------------------------------------------------

    def _read_field(
        self,
        collection: str,
        asset_key: str,
        cf_variable: str,
        pressure_hpa: int | None,
    ) -> np.ndarray:
        """Read a 2-D field from a cached Met Office NetCDF file.

        Datasets are cached in ``self._ds_cache`` for the duration of a
        single timestamp's extraction (cleared in :meth:`_fetch_data`).

        Parameters
        ----------
        collection : str
            STAC collection identifier.
        asset_key : str
            Asset key within the STAC item.
        cf_variable : str
            CF variable name inside the NetCDF file.
        pressure_hpa : int or None
            Pressure level in hPa, or ``None`` for surface fields.

        Returns
        -------
        np.ndarray
            Raw 2-D array on the native Met Office grid.
        """
        # Resolve local path from the asset plan cache
        cache_dir = pathlib.Path(self.cache)
        # We need to find the local path for this asset. Build the unsigned
        # href pattern from _locate_item results stored in _current_items.
        item = self._current_items[collection]
        asset = item.assets[asset_key]
        local_path = self._local_asset_path(asset.href)

        cache_key = str(local_path)
        if cache_key not in self._ds_cache:
            self._ds_cache[cache_key] = xr.open_dataset(local_path, engine="h5netcdf")

        ds = self._ds_cache[cache_key]
        field = ds[cf_variable]

        if pressure_hpa is not None and "pressure" in field.dims:
            target_pa = _HPA_TO_PA[pressure_hpa]
            field = field.sel(pressure=target_pa, method="nearest")

        return field.values.squeeze().astype(np.float32)

    # ------------------------------------------------------------------
    # Override _fetch_data to manage the dataset cache lifecycle
    # ------------------------------------------------------------------

    async def _fetch_data(
        self,
        client: Any,
        semaphore: Any,
        requested_time: datetime,
        variables: Sequence[VariableSpec],
        xr_array: xr.DataArray,
        time_index: int,
        progress: Any,
    ) -> None:
        """Download and extract all variables for one timestamp.

        Wraps the base-class pipeline to manage the per-timestamp dataset
        cache and to store the current STAC items for :meth:`_read_field`.
        """
        import asyncio as _asyncio

        # Locate items for both collections.
        items = await _asyncio.to_thread(self._locate_item, requested_time)
        self._current_items = items  # used by _read_field

        # Build and execute asset plans.
        asset_plans = self._prepare_asset_plans(items, variables)
        download_tasks = [
            self._downloaded_asset(client, semaphore, plan)
            for plan in asset_plans
            if not plan.local_path.exists()
        ]
        if download_tasks:
            await _asyncio.gather(*download_tasks)

        # Collect which spec indices are covered by plans.
        assigned_indices: set[int] = set()
        for plan in asset_plans:
            for spec in plan.variables:
                assigned_indices.add(spec.index)

        # Extract all variables with dataset caching.
        self._ds_cache.clear()
        try:
            data_stack = np.zeros(
                (len(variables), *self._spatial_shape), dtype=np.float32,
            )
            # Handle plan-assigned variables.
            for plan in asset_plans:
                for spec in plan.variables:
                    data_stack[spec.index] = self.extract_variable_numpy(
                        plan, spec, requested_time,
                    )
            # Handle variables with no assets (e.g. constant fills).
            for spec in variables:
                if spec.index not in assigned_indices:
                    data_stack[spec.index] = self.extract_variable_numpy(
                        None, spec, requested_time,  # type: ignore[arg-type]
                    )
            xr_array[time_index] = data_stack
        finally:
            for ds in self._ds_cache.values():
                ds.close()
            self._ds_cache.clear()
            self._current_items = {}  # type: ignore[assignment]

        progress.update(len(variables))

    # ------------------------------------------------------------------
    # Regridding
    # ------------------------------------------------------------------

    def _regrid(self, data: np.ndarray) -> np.ndarray:
        """Bilinear regrid from the native Met Office grid to the Atlas grid.

        Handles:
        1. Longitude shift from [-180, 180] → [0, 360)
        2. Latitude flip from S→N (Met Office) to N→S (Atlas)
        3. ``scipy.interpolate.RegularGridInterpolator`` for efficiency

        The interpolator grid metadata is lazily computed and cached on the
        instance for reuse across variables.
        """
        from scipy.interpolate import RegularGridInterpolator

        nlat_src, nlon_src = data.shape

        if (
            self._regrid_state is None
            or self._regrid_state["shape"] != (nlat_src, nlon_src)
        ):
            lat_src = np.linspace(-89.953125, 89.953125, nlat_src, dtype=np.float64)
            lon_src = np.linspace(-179.9296875, 179.9296875, nlon_src, dtype=np.float64)

            lon360 = lon_src % 360.0
            sort_idx = np.argsort(lon360)
            lon_sorted = lon360[sort_idx]
            lat_desc = lat_src[::-1]  # N→S

            target_lat = self.LAT_COORDS.astype(np.float64)
            target_lon = self.LON_COORDS.astype(np.float64)

            self._regrid_state = {
                "shape": (nlat_src, nlon_src),
                "sort_idx": sort_idx,
                "lat_desc": lat_desc,
                "lon_sorted": lon_sorted,
                "target_lat": target_lat,
                "target_lon": target_lon,
            }

        st = self._regrid_state
        data_shifted = data[:, st["sort_idx"]][::-1, :]

        interp = RegularGridInterpolator(
            (st["lat_desc"], st["lon_sorted"]),
            data_shifted.astype(np.float64),
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate at poles
        )

        target_grid = np.stack(
            np.meshgrid(st["target_lat"], st["target_lon"], indexing="ij"),
            axis=-1,
        )
        return interp(target_grid).astype(np.float32)

    # ------------------------------------------------------------------
    # Cache path
    # ------------------------------------------------------------------

    @property
    def cache(self) -> str:
        """Local cache directory for Met Office assets."""
        cache_root = os.path.join(
            datasource_cache_root(), "planetary_computer", "met_office",
        )
        if not self._cache:
            cache_root = os.path.join(cache_root, "tmp")
        return cache_root
