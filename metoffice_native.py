# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""Pure data loader for Met Office global deterministic forecast data from
Microsoft Planetary Computer.

This module provides :class:`PlanetaryComputerMetOfficeNative`, a DataSource
that returns raw Met Office fields on the **native** ~0.09° grid using
native Met Office variable names.  No derived variables, no regridding,
longitude normalized to [0, 360].

Variable names follow the pattern:

- Surface fields: ``air_temperature_at_screen_level``,
  ``air_pressure_at_sea_level``, ``wind_speed_at_10m``,
  ``wind_from_direction_at_10m``, ``surface_temperature``,
  ``lwe_precipitation_rate``
- Pressure-level fields: ``air_temperature_500hPa``,
  ``wind_speed_850hPa``, ``relative_humidity_200hPa``,
  ``geopotential_height_1000hPa``, etc.

The output xarray DataArray has:
- lat: ascending (S→N), native resolution (~0.09°)
- lon: [0, 360] (converted from native [-180, 180])
- No derived variables — just what's in the NetCDF files

Usage::

    from metoffice_native import PlanetaryComputerMetOfficeNative
    from datetime import datetime

    ds = PlanetaryComputerMetOfficeNative()
    da = ds(datetime(2026, 2, 17), ["air_temperature_500hPa", "wind_speed_at_10m"])
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.data.planetary_computer import (
    AssetPlan,
    VariableSpec,
    _PlanetaryComputerData,
)
from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.base import LexiconType

try:
    import planetary_computer
    from pystac_client import Client
except ImportError:
    planetary_computer = None
    Client = None

PRESSURE_LEVELS_HPA: list[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]

# Met Office NetCDF files use Pa for the pressure coordinate
_HPA_TO_PA: dict[int, float] = {
    hpa: float(hpa * 100) for hpa in PRESSURE_LEVELS_HPA
}

_PRESSURE_ASSETS: dict[str, str] = {
    "geopotential_height_on_pressure_levels": "geopotential_height",
    "temperature_on_pressure_levels": "air_temperature",
    "wind_speed_on_pressure_levels": "wind_speed",
    "wind_direction_on_pressure_levels": "wind_from_direction",
    "relative_humidity_on_pressure_levels": "relative_humidity",
}

_SURFACE_ASSETS: dict[str, str] = {
    "temperature_at_screen_level": "air_temperature",
    "pressure_at_mean_sea_level": "air_pressure_at_sea_level",
    "wind_speed_at_10m": "wind_speed",
    "wind_direction_at_10m": "wind_from_direction",
    "temperature_at_surface": "surface_temperature",
    "precipitation_rate": "lwe_precipitation_rate",
}

#: Callable that post-processes a raw field (identity for native variables).
Modifier = Callable[[Any], Any]


def _nmod(x: Any) -> Any:
    return x


# CF names alone are ambiguous (e.g. "air_temperature" appears at both
# screen level and on pressure levels), so surface variables use descriptive
# names derived from the STAC asset keys.
_ASSET_KEY_TO_NATIVE_NAME: dict[str, str] = {
    "temperature_at_screen_level": "air_temperature_at_screen_level",
    "pressure_at_mean_sea_level": "air_pressure_at_sea_level",
    "wind_speed_at_10m": "wind_speed_at_10m",
    "wind_direction_at_10m": "wind_from_direction_at_10m",
    "temperature_at_surface": "surface_temperature",
    "precipitation_rate": "lwe_precipitation_rate",
}


def _surface_variable_name(asset_key: str, cf_var: str) -> str:
    """Canonical variable name for a surface field."""
    return _ASSET_KEY_TO_NATIVE_NAME.get(asset_key, f"{cf_var}__{asset_key}")


def _pressure_variable_name(cf_var: str, hpa: int) -> str:
    return f"{cf_var}_{hpa}hPa"


def _all_native_variables() -> list[str]:
    """All native Met Office variable names (surface first, then pressure-level)."""
    names: list[str] = []
    for asset_key, cf_var in _SURFACE_ASSETS.items():
        names.append(_surface_variable_name(asset_key, cf_var))
    for _asset_key, cf_var in _PRESSURE_ASSETS.items():
        for hpa in PRESSURE_LEVELS_HPA:
            names.append(_pressure_variable_name(cf_var, hpa))
    return names


ALL_NATIVE_VARIABLES: list[str] = _all_native_variables()


def _build_native_vocab() -> dict[str, tuple[str, Modifier]]:
    """Build the vocabulary for native Met Office variable names.

    Dataset keys use the format ``<collection_type>::<asset_key>::<cf_variable>[::pressure_hPa]``.
    """
    vocab: dict[str, tuple[str, Modifier]] = {}

    for asset_key, cf_var in _SURFACE_ASSETS.items():
        var_name = _surface_variable_name(asset_key, cf_var)
        dataset_key = f"surface::{asset_key}::{cf_var}"
        vocab[var_name] = (dataset_key, _nmod)

    for asset_key, cf_var in _PRESSURE_ASSETS.items():
        for hpa in PRESSURE_LEVELS_HPA:
            var_name = _pressure_variable_name(cf_var, hpa)
            dataset_key = f"pressure::{asset_key}::{cf_var}::{hpa}"
            vocab[var_name] = (dataset_key, _nmod)

    return vocab


class MetOfficeNativeLexicon(metaclass=LexiconType):
    """Lexicon mapping native Met Office variable names to STAC asset locations."""

    VOCAB = _build_native_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        return cls.VOCAB[val]


# Native ~0.09° grid: 1920 lats (S→N), 2560 lons
NATIVE_NLAT: int = 1920
NATIVE_NLON: int = 2560

NATIVE_LAT_COORDS = np.linspace(-89.953125, 89.953125, NATIVE_NLAT, dtype=np.float32)

# Raw NetCDF files use [-180, 180] but Atlas (and most models) need [0, 360].
# We roll the data in _read_field and declare coords in [0, 360] so regridding
# finds valid data across the full domain.
_RAW_LON_COORDS = np.linspace(
    -179.9296875, 179.9296875, NATIVE_NLON, dtype=np.float32
)
NATIVE_LON_COORDS = _RAW_LON_COORDS % 360
NATIVE_LON_COORDS.sort()
# Number of grid points to roll: shift the 0° meridian to index 0
_LON_ROLL = int(np.searchsorted(_RAW_LON_COORDS, 0.0))


class PlanetaryComputerMetOfficeNative(_PlanetaryComputerData):
    """Earth2Studio DataSource for raw Met Office Global 10 km Deterministic
    forecast data from Microsoft Planetary Computer.

    Returns fields on the **native** Met Office grid (~0.09° resolution)
    with native variable names.  No derived variables, no regridding,
    longitude normalized to [0, 360].

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

    Example
    -------
    >>> from metoffice_native import PlanetaryComputerMetOfficeNative
    >>> from datetime import datetime
    >>> ds = PlanetaryComputerMetOfficeNative()
    >>> da = ds(datetime(2026, 2, 17), ["air_temperature_500hPa", "wind_speed_at_10m"])
    """

    PRESSURE_COLLECTION = "met-office-global-deterministic-pressure"
    SURFACE_COLLECTION = "met-office-global-deterministic-near-surface"

    LAT_COORDS = NATIVE_LAT_COORDS
    LON_COORDS = NATIVE_LON_COORDS

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
        super().__init__(
            collection_id=self.PRESSURE_COLLECTION,
            lexicon=MetOfficeNativeLexicon,
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
        self._ds_cache: dict[str, xr.Dataset] = {}
        self._current_items: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Time validation & fetch override
    # ------------------------------------------------------------------
    async def fetch(
        self,
        time: datetime | list[datetime],
        variable: str | list[str],
    ) -> xr.DataArray:
        """Validate times then delegate to the base-class fetch pipeline."""
        times, _ = prep_data_inputs(time, variable)
        self._validate_time(times)
        return await super().fetch(time, variable)

    @staticmethod
    def _validate_time(times: list[datetime]) -> None:
        """Verify requested times fall on 6-hourly Met Office run boundaries."""
        for t in times:
            if (
                t.hour not in PlanetaryComputerMetOfficeNative._VALID_RUN_HOURS
                or t.minute
                or t.second
            ):
                raise ValueError(
                    f"Met Office reference time {t} must be on a 6-hour boundary "
                    f"(00, 06, 12, or 18 UTC)"
                )

    # ------------------------------------------------------------------
    # STAC search
    # ------------------------------------------------------------------
    def _locate_item(self, when: datetime) -> dict[str, Any]:
        """Locate STAC items for *both* collections at the given reference time."""
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
    # Asset planning
    # ------------------------------------------------------------------
    def _prepare_asset_plans(
        self,
        items: Any,  # dict[str, pystac.Item]
        variables: Sequence[VariableSpec],
    ) -> list[AssetPlan]:
        """Build download plans for all assets required by *variables*.

        Each unique (collection, asset_key) pair produces one AssetPlan.
        Each VariableSpec is assigned to exactly one plan.
        """
        needed_assets: dict[tuple[str, str], list[VariableSpec]] = {}

        for spec in variables:
            col, asset_key = self._spec_to_asset_key(spec)
            key = (col, asset_key)
            needed_assets.setdefault(key, []).append(spec)

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
    def _spec_to_asset_key(spec: VariableSpec) -> tuple[str, str]:
        """Map a variable spec to its (collection_id, asset_key)."""
        parts = spec.dataset_key.split("::")
        collection_type = parts[0]
        asset_key = parts[1]
        col = (
            PlanetaryComputerMetOfficeNative.PRESSURE_COLLECTION
            if collection_type == "pressure"
            else PlanetaryComputerMetOfficeNative.SURFACE_COLLECTION
        )
        return col, asset_key

    # ------------------------------------------------------------------
    # Variable extraction
    # ------------------------------------------------------------------
    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        _target_time: datetime,
    ) -> np.ndarray:
        """Extract a single variable as a raw native-grid field."""
        parts = spec.dataset_key.split("::")
        cf_var = parts[2]
        pressure_hpa = int(parts[3]) if len(parts) > 3 else None

        raw = self._read_field(plan.local_path, cf_var, pressure_hpa)
        return spec.modifier(raw)

    def _read_field(
        self,
        local_path: pathlib.Path,
        cf_variable: str,
        pressure_hpa: int | None,
    ) -> np.ndarray:
        """Read a 2-D field from a cached Met Office NetCDF file."""
        cache_key = str(local_path)
        if cache_key not in self._ds_cache:
            self._ds_cache[cache_key] = xr.open_dataset(
                local_path, engine="h5netcdf"
            )

        ds = self._ds_cache[cache_key]
        field = ds[cf_variable]

        if pressure_hpa is not None and "pressure" in field.dims:
            target_pa = _HPA_TO_PA[pressure_hpa]
            field = field.sel(pressure=target_pa, method="nearest")

        arr = field.values.squeeze().astype(np.float32)
        # Roll lon from [-180,180] → [0,360] to match NATIVE_LON_COORDS
        return np.roll(arr, -_LON_ROLL, axis=-1)

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
        """Download and extract all variables for one timestamp."""
        import asyncio as _asyncio

        items = await _asyncio.to_thread(self._locate_item, requested_time)
        self._current_items = items

        asset_plans = self._prepare_asset_plans(items, variables)
        download_tasks = [
            self._downloaded_asset(client, semaphore, plan)
            for plan in asset_plans
            if not plan.local_path.exists()
        ]
        if download_tasks:
            await _asyncio.gather(*download_tasks)

        self._ds_cache.clear()
        try:
            data_stack = np.zeros(
                (len(variables), *self._spatial_shape), dtype=np.float32,
            )
            for plan in asset_plans:
                for spec in plan.variables:
                    data_stack[spec.index] = self.extract_variable_numpy(
                        plan, spec, requested_time,
                    )
            xr_array[time_index] = data_stack
        finally:
            for ds in self._ds_cache.values():
                ds.close()
            self._ds_cache.clear()
            self._current_items = {}

        progress.update(len(variables))

    @property
    def cache(self) -> str:
        """Local cache directory for Met Office assets."""
        cache_root = os.path.join(
            datasource_cache_root(), "planetary_computer", "met_office",
        )
        if not self._cache:
            cache_root = os.path.join(cache_root, "tmp")
        return cache_root
