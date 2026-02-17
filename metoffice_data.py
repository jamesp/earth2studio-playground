# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""Custom Earth2Studio data source for Met Office global deterministic forecast
data hosted on Microsoft Planetary Computer.

This module provides a `DataSource`-compatible class that fetches pressure-level and
near-surface fields from two Met Office STAC collections, performs the necessary
transformations (wind speed/direction → u/v components, relative humidity → specific
humidity, geopotential height → geopotential, regridding from ~0.09° to 0.25°,
longitude shift from [-180,180] to [0,360]), and returns data on the 721×1440 grid
expected by NVIDIA Atlas.

Usage::

    from metoffice_data import PlanetaryComputerMetOffice
    from datetime import datetime

    ds = PlanetaryComputerMetOffice()
    da = ds(datetime(2026, 2, 17), ["t2m", "u500", "z500", "msl"])
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import nest_asyncio
import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.base import LexiconType
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import httpx
    import planetary_computer
    from pystac_client import Client
except ImportError:
    httpx = None
    Client = None
    planetary_computer = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Atlas grid: 0.25° global, 721 lat × 1440 lon, lon in [0, 360)
ATLAS_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float32)
ATLAS_LON = np.linspace(0.0, 360.0, 1440, dtype=np.float32, endpoint=False)

# Pressure levels required by Atlas (hPa)
ATLAS_PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Mapping from Atlas hPa to Met Office Pa for pressure coordinate selection
_HPA_TO_PA = {hpa: float(hpa * 100) for hpa in ATLAS_PRESSURE_LEVELS_HPA}

# Gravity for geopotential conversion (m/s²)
G = np.float32(9.80665)

# Constants for specific humidity from relative humidity
# Using the Tetens/Bolton formula: e_s(T) = 611.2 * exp(17.67*(T-273.15)/(T-29.65))
# q = (epsilon * e) / (p - (1-epsilon)*e)  where e = rh * e_s
EPSILON = np.float32(0.622)  # ratio of molecular weight of water to dry air


def _saturation_vapor_pressure(t_k: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure (Pa) from temperature (K) using Bolton (1980)."""
    t_c = t_k - np.float32(273.15)
    return np.float32(611.2) * np.exp(
        np.float32(17.67) * t_c / (t_c + np.float32(243.5))
    )


def _specific_humidity(
    rh_frac: np.ndarray, t_k: np.ndarray, p_pa: np.ndarray
) -> np.ndarray:
    """Compute specific humidity (kg/kg) from fractional RH, T (K), and P (Pa)."""
    e_s = _saturation_vapor_pressure(t_k)
    e = rh_frac * e_s
    q = EPSILON * e / (p_pa - (np.float32(1.0) - EPSILON) * e)
    return np.clip(q, 0.0, None).astype(np.float32)


def _wind_components(
    speed: np.ndarray, direction_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind speed and direction (degrees, 'from') to u, v.

    Met convention: direction is where the wind is blowing FROM, measured
    clockwise from north.
      u = -speed * sin(direction)
      v = -speed * cos(direction)
    """
    direction_rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)
    return u.astype(np.float32), v.astype(np.float32)


# ---------------------------------------------------------------------------
# Lexicon: maps Earth2Studio variable names to extraction specifications.
#
# Format: e2s_variable -> (spec_string, modifier_fn)
# spec_string encodes: collection::asset_key::cf_variable_name[::pressure_hPa]
#   collection: "pressure" or "surface"
#   For derived variables (u/v wind components, specific humidity), we use
#   special pseudo-collection prefixes like "derive_uv_pressure" etc.
# ---------------------------------------------------------------------------

Modifier = Callable[[Any], Any]
IDENTITY: Modifier = lambda x: x


def _build_vocab() -> dict[str, tuple[str, Modifier]]:
    """Build the full lexicon mapping for Met Office → Earth2Studio variables."""
    vocab: dict[str, tuple[str, Modifier]] = {}

    # --- Near-surface variables ---
    # t2m: screen-level temperature (1.5m, close enough to 2m)
    vocab["t2m"] = ("surface::temperature_at_screen_level::air_temperature", IDENTITY)

    # sp: surface pressure — Met Office provides MSLP but not surface pressure directly.
    # We use MSLP as an approximation. A proper implementation would compute sp from
    # MSLP, temperature, and orographic height, but that requires additional fields.
    vocab["sp"] = (
        "surface::pressure_at_mean_sea_level::air_pressure_at_sea_level",
        IDENTITY,
    )

    # msl: mean sea level pressure (Pa)
    vocab["msl"] = (
        "surface::pressure_at_mean_sea_level::air_pressure_at_sea_level",
        IDENTITY,
    )

    # u10m, v10m: derived from wind speed + direction at 10m
    vocab["u10m"] = ("derive_uv_surface_10m::u", IDENTITY)
    vocab["v10m"] = ("derive_uv_surface_10m::v", IDENTITY)

    # u100m, v100m: Met Office doesn't provide 100m winds.
    # As a fallback, use 10m winds as an approximation.
    vocab["u100m"] = ("derive_uv_surface_10m::u", IDENTITY)
    vocab["v100m"] = ("derive_uv_surface_10m::v", IDENTITY)

    # tcwv: total column water vapour — not available from Met Office.
    # Fill with zeros; this is a known limitation.
    vocab["tcwv"] = ("constant::0.0", IDENTITY)

    # sst: sea surface temperature — use surface temperature as proxy
    vocab["sst"] = ("surface::temperature_at_surface::surface_temperature", IDENTITY)

    # tp: total precipitation — use instantaneous precip rate as proxy (will be ~0 at T+0)
    vocab["tp"] = ("surface::precipitation_rate::lwe_precipitation_rate", IDENTITY)

    # --- Pressure-level variables ---
    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        # u-wind component at pressure level: derived from speed + direction
        vocab[f"u{hpa}"] = (f"derive_uv_pressure::{hpa}::u", IDENTITY)
        # v-wind component at pressure level
        vocab[f"v{hpa}"] = (f"derive_uv_pressure::{hpa}::v", IDENTITY)

        # Geopotential: Met Office provides geopotential HEIGHT (m),
        # Atlas expects geopotential (m² s⁻²) = height × g
        vocab[f"z{hpa}"] = (
            f"pressure::geopotential_height_on_pressure_levels::geopotential_height::{hpa}",
            lambda x: x * G,
        )

        # Temperature at pressure level (K) — direct
        vocab[f"t{hpa}"] = (
            f"pressure::temperature_on_pressure_levels::air_temperature::{hpa}",
            IDENTITY,
        )

        # Specific humidity at pressure level: derived from RH + T + P
        vocab[f"q{hpa}"] = (f"derive_q_pressure::{hpa}", IDENTITY)

    return vocab


class MetOfficeLexicon(metaclass=LexiconType):
    """Lexicon mapping Earth2Studio variable names to Met Office data specifications."""

    VOCAB = _build_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Modifier]:
        return cls.VOCAB[val]


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------


class PlanetaryComputerMetOffice:
    """Earth2Studio DataSource for Met Office Global 10km Deterministic forecast
    from Microsoft Planetary Computer.

    Fetches T+0 (analysis-like) forecast fields from the Met Office global model
    and returns them on the 0.25° 721×1440 grid used by NVIDIA Atlas.

    Handles:
    - Wind speed + direction → u/v component decomposition
    - Relative humidity → specific humidity conversion
    - Geopotential height (m) → geopotential (m² s⁻²) conversion
    - Regridding from ~0.09° to 0.25° (bilinear interpolation)
    - Longitude convention shift from [-180, 180] to [0, 360)

    Parameters
    ----------
    forecast_hour : int, optional
        Forecast lead time in hours to retrieve, by default 0 (T+0).
    cache : bool, optional
        Cache downloaded files locally, by default True.
    verbose : bool, optional
        Print progress information, by default True.
    request_timeout : int, optional
        HTTP request timeout in seconds, by default 120.
    max_retries : int, optional
        Maximum HTTP retry attempts, by default 4.
    max_workers : int, optional
        Maximum concurrent downloads, by default 8.

    Note
    ----
    Known limitations:
    - ``tcwv`` (total column water vapour) is not available and is filled with zeros.
    - ``u100m``/``v100m`` (100m wind) fall back to 10m winds.
    - ``sp`` (surface pressure) uses MSLP as an approximation.

    Example
    -------
    >>> from metoffice_data import PlanetaryComputerMetOffice
    >>> from datetime import datetime
    >>> ds = PlanetaryComputerMetOffice()
    >>> da = ds(datetime(2026, 2, 17), ["t2m", "u500", "z500"])
    """

    PRESSURE_COLLECTION = "met-office-global-deterministic-pressure"
    SURFACE_COLLECTION = "met-office-global-deterministic-near-surface"
    CHUNK_SIZE = 1 << 20
    USER_AGENT = "earth2studio-metoffice"

    def __init__(
        self,
        forecast_hour: int = 0,
        cache: bool = True,
        verbose: bool = True,
        request_timeout: int = 120,
        max_retries: int = 4,
        max_workers: int = 8,
    ) -> None:
        self._forecast_hour = forecast_hour
        self._cache = cache
        self._verbose = verbose
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._max_workers = max_workers
        self._lexicon = MetOfficeLexicon
        self._client: Client | None = None

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch Met Office data for the given times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC). These are treated as forecast
            reference times (i.e. model run times).
        variable : str | list[str] | VariableArray
            Earth2Studio variable names (e.g. "t2m", "u500", "z500").

        Returns
        -------
        xr.DataArray
            Data on the Atlas grid [time, variable, lat, lon].
        """
        nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(self.fetch(time, variable))
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)
        return result

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async fetch implementation."""
        times, variables = prep_data_inputs(time, variable)

        normalized_times = [
            (
                t.replace(tzinfo=timezone.utc)
                if t.tzinfo is None
                else t.astimezone(timezone.utc)
            )
            for t in times
        ]

        # Create cache directory
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Pre-allocate output array on the Atlas grid
        xr_array = xr.DataArray(
            data=np.zeros(
                (len(times), len(variables), len(ATLAS_LAT), len(ATLAS_LON)),
                dtype=np.float32,
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": np.array(
                    [np.datetime64(t.replace(tzinfo=None)) for t in normalized_times]
                ),
                "variable": list(variables),
                "lat": ATLAS_LAT,
                "lon": ATLAS_LON,
            },
        )

        timeout = httpx.Timeout(self._request_timeout)
        limits = httpx.Limits(max_connections=self._max_workers)
        transport = httpx.AsyncHTTPTransport(limits=limits, retries=self._max_retries)
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            semaphore = asyncio.Semaphore(self._max_workers)
            with tqdm(
                total=len(times),
                disable=not self._verbose,
                desc="Fetching Met Office data",
            ) as progress:
                tasks = [
                    asyncio.create_task(
                        self._fetch_time(
                            http_client=client,
                            semaphore=semaphore,
                            requested_time=normalized_times[i],
                            variables=variables,
                            xr_array=xr_array,
                            time_index=i,
                            progress=progress,
                        )
                    )
                    for i in range(len(times))
                ]
                if tasks:
                    await asyncio.gather(*tasks)

        return xr_array

    async def _fetch_time(
        self,
        http_client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        requested_time: datetime,
        variables: list[str],
        xr_array: xr.DataArray,
        time_index: int,
        progress: tqdm,
    ) -> None:
        """Fetch all variables for a single timestamp."""
        # Determine which raw assets we need
        plan = self._plan_downloads(variables)

        # Locate STAC items for both collections
        items = await asyncio.to_thread(
            self._locate_items, requested_time, plan
        )

        # Download all required assets
        download_tasks = []
        for (collection, asset_key), info in plan["assets"].items():
            if (collection, asset_key) not in items:
                continue
            item = items[(collection, asset_key)]
            asset = item.assets[asset_key]
            signed = planetary_computer.sign(asset)
            local_path = self._local_path(asset.href)
            if not local_path.exists():
                download_tasks.append(
                    self._download_asset(
                        http_client, semaphore, signed.href, local_path
                    )
                )
            info["local_path"] = local_path
            info["item"] = item

        if download_tasks:
            await asyncio.gather(*download_tasks)

        # Extract and regrid each variable
        # Use a dataset cache to avoid re-opening large NetCDF files
        plan["_ds_cache"] = {}
        data_stack = np.zeros(
            (len(variables), len(ATLAS_LAT), len(ATLAS_LON)), dtype=np.float32
        )
        try:
            for i, var in enumerate(variables):
                data_stack[i] = self._extract_variable(var, plan)
        finally:
            # Close all cached datasets
            for ds in plan["_ds_cache"].values():
                ds.close()
            del plan["_ds_cache"]

        xr_array[time_index] = data_stack
        progress.update(1)

    def _plan_downloads(
        self, variables: list[str]
    ) -> dict[str, Any]:
        """Determine which NetCDF assets need to be downloaded for the requested variables.

        Returns a plan dict with:
        - 'assets': {(collection, asset_key): {'local_path': ..., 'needed_for': [...]}}
        - 'variable_specs': {var_name: (spec_string, modifier)}
        """
        plan: dict[str, Any] = {
            "assets": {},
            "variable_specs": {},
        }

        for var in variables:
            spec_str, modifier = self._lexicon[var]
            plan["variable_specs"][var] = (spec_str, modifier)

            # Determine which raw assets this variable needs
            needed_assets = self._spec_to_assets(spec_str)
            for collection, asset_key in needed_assets:
                key = (collection, asset_key)
                if key not in plan["assets"]:
                    plan["assets"][key] = {
                        "local_path": None,
                        "item": None,
                        "needed_for": [],
                    }
                plan["assets"][key]["needed_for"].append(var)

        return plan

    def _spec_to_assets(
        self, spec_str: str
    ) -> list[tuple[str, str]]:
        """Given a spec string, return list of (collection, asset_key) needed."""
        if spec_str.startswith("constant::"):
            return []

        if spec_str.startswith("derive_uv_surface_10m::"):
            return [
                (self.SURFACE_COLLECTION, "wind_speed_at_10m"),
                (self.SURFACE_COLLECTION, "wind_direction_at_10m"),
            ]

        if spec_str.startswith("derive_uv_pressure::"):
            return [
                (self.PRESSURE_COLLECTION, "wind_speed_on_pressure_levels"),
                (self.PRESSURE_COLLECTION, "wind_direction_on_pressure_levels"),
            ]

        if spec_str.startswith("derive_q_pressure::"):
            return [
                (self.PRESSURE_COLLECTION, "temperature_on_pressure_levels"),
                (self.PRESSURE_COLLECTION, "relative_humidity_on_pressure_levels"),
            ]

        parts = spec_str.split("::")
        collection_type = parts[0]  # "pressure" or "surface"
        asset_key = parts[1]

        if collection_type == "pressure":
            return [(self.PRESSURE_COLLECTION, asset_key)]
        elif collection_type == "surface":
            return [(self.SURFACE_COLLECTION, asset_key)]
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")

    def _locate_items(
        self, when: datetime, plan: dict[str, Any]
    ) -> dict[tuple[str, str], Any]:
        """Locate STAC items for both collections matching the requested time."""
        if self._client is None:
            self._client = Client.open(STAC_API_URL)

        ref_time_str = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        horizon_str = f"PT{self._forecast_hour:04d}H00M"

        items = {}
        collections_needed = set()
        for (collection, asset_key) in plan["assets"]:
            collections_needed.add(collection)

        for collection in collections_needed:
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
                item = next(search.items())
            except StopIteration:
                raise FileNotFoundError(
                    f"No Met Office item found for ref={ref_time_str} "
                    f"horizon={horizon_str} in {collection}"
                )
            # Map this item for all asset keys from this collection
            for (col, asset_key) in plan["assets"]:
                if col == collection:
                    items[(col, asset_key)] = item

        return items

    def _extract_variable(
        self, var: str, plan: dict[str, Any]
    ) -> np.ndarray:
        """Extract a single variable from cached NetCDF files and regrid to Atlas grid.

        Returns a (721, 1440) float32 array.

        Uses plan['_ds_cache'] to avoid re-opening large NetCDF files.
        """
        spec_str, modifier = plan["variable_specs"][var]

        if spec_str.startswith("constant::"):
            value = float(spec_str.split("::")[1])
            return np.full((len(ATLAS_LAT), len(ATLAS_LON)), value, dtype=np.float32)

        if spec_str.startswith("derive_uv_surface_10m::"):
            component = spec_str.split("::")[1]  # "u" or "v"
            return self._extract_uv_surface(component, plan)

        if spec_str.startswith("derive_uv_pressure::"):
            parts = spec_str.split("::")
            hpa = int(parts[1])
            component = parts[2]  # "u" or "v"
            return self._extract_uv_pressure(hpa, component, plan)

        if spec_str.startswith("derive_q_pressure::"):
            hpa = int(spec_str.split("::")[1])
            return self._extract_q_pressure(hpa, plan)

        # Direct extraction from a single NetCDF asset
        parts = spec_str.split("::")
        collection_type = parts[0]
        asset_key = parts[1]
        cf_var = parts[2]
        pressure_hpa = int(parts[3]) if len(parts) > 3 else None

        if collection_type == "pressure":
            collection = self.PRESSURE_COLLECTION
        else:
            collection = self.SURFACE_COLLECTION

        local_path = plan["assets"][(collection, asset_key)]["local_path"]
        ds_cache = plan.get("_ds_cache")
        raw = self._read_netcdf_variable(local_path, cf_var, pressure_hpa, ds_cache)
        raw = modifier(raw)
        return self._regrid(raw)

    def _extract_uv_surface(
        self, component: str, plan: dict[str, Any]
    ) -> np.ndarray:
        """Extract u or v wind at 10m from speed + direction."""
        speed_path = plan["assets"][
            (self.SURFACE_COLLECTION, "wind_speed_at_10m")
        ]["local_path"]
        dir_path = plan["assets"][
            (self.SURFACE_COLLECTION, "wind_direction_at_10m")
        ]["local_path"]

        ds_cache = plan.get("_ds_cache")
        speed = self._read_netcdf_variable(speed_path, "wind_speed", None, ds_cache)
        direction = self._read_netcdf_variable(dir_path, "wind_from_direction", None, ds_cache)
        u, v = _wind_components(speed, direction)

        if component == "u":
            return self._regrid(u)
        else:
            return self._regrid(v)

    def _extract_uv_pressure(
        self, hpa: int, component: str, plan: dict[str, Any]
    ) -> np.ndarray:
        """Extract u or v wind at a pressure level from speed + direction."""
        speed_path = plan["assets"][
            (self.PRESSURE_COLLECTION, "wind_speed_on_pressure_levels")
        ]["local_path"]
        dir_path = plan["assets"][
            (self.PRESSURE_COLLECTION, "wind_direction_on_pressure_levels")
        ]["local_path"]

        ds_cache = plan.get("_ds_cache")
        speed = self._read_netcdf_variable(
            speed_path, "wind_speed", hpa, ds_cache
        )
        direction = self._read_netcdf_variable(
            dir_path, "wind_from_direction", hpa, ds_cache
        )
        u, v = _wind_components(speed, direction)

        if component == "u":
            return self._regrid(u)
        else:
            return self._regrid(v)

    def _extract_q_pressure(
        self, hpa: int, plan: dict[str, Any]
    ) -> np.ndarray:
        """Derive specific humidity from RH + T at a pressure level."""
        temp_path = plan["assets"][
            (self.PRESSURE_COLLECTION, "temperature_on_pressure_levels")
        ]["local_path"]
        rh_path = plan["assets"][
            (self.PRESSURE_COLLECTION, "relative_humidity_on_pressure_levels")
        ]["local_path"]

        ds_cache = plan.get("_ds_cache")
        t_k = self._read_netcdf_variable(temp_path, "air_temperature", hpa, ds_cache)
        rh_frac = self._read_netcdf_variable(rh_path, "relative_humidity", hpa, ds_cache)

        p_pa = np.float32(hpa * 100)
        q = _specific_humidity(rh_frac, t_k, p_pa)
        return self._regrid(q)

    def _read_netcdf_variable(
        self,
        path: pathlib.Path,
        cf_variable: str,
        pressure_hpa: int | None,
        ds_cache: dict | None = None,
    ) -> np.ndarray:
        """Read a 2D field from a Met Office NetCDF file.

        For pressure-level files, selects the requested level.
        Returns the raw (lat, lon) array on the native Met Office grid.

        If ds_cache is provided, caches opened datasets to avoid repeated I/O.
        """
        cache_key = str(path)
        if ds_cache is not None and cache_key in ds_cache:
            ds = ds_cache[cache_key]
        else:
            ds = xr.open_dataset(path, engine="h5netcdf")
            if ds_cache is not None:
                ds_cache[cache_key] = ds

        field = ds[cf_variable]
        if pressure_hpa is not None and "pressure" in field.dims:
            # Pressure coordinate is in Pa in the file
            target_pa = _HPA_TO_PA[pressure_hpa]
            field = field.sel(pressure=target_pa, method="nearest")
        values = field.values.astype(np.float32)

        # Close if not caching
        if ds_cache is None:
            ds.close()

        # Squeeze any remaining scalar dimensions
        values = values.squeeze()
        return values

    _regrid_state: dict | None = None  # Class-level cache for regrid weights

    def _regrid(self, data: np.ndarray) -> np.ndarray:
        """Regrid a (1920, 2560) Met Office field to the (721, 1440) Atlas grid.

        Steps:
        1. Build source lat/lon from the Met Office grid
        2. Shift longitude from [-180, 180] → [0, 360)
        3. Bilinear interpolation to Atlas grid using scipy

        Uses cached interpolation weights for efficiency.
        """
        from scipy.interpolate import RegularGridInterpolator

        nlat_src, nlon_src = data.shape

        # Build and cache the source grid metadata
        if self._regrid_state is None or self._regrid_state["shape"] != (nlat_src, nlon_src):
            lat_src = np.linspace(-89.953125, 89.953125, nlat_src, dtype=np.float64)
            lon_src = np.linspace(-179.9296875, 179.9296875, nlon_src, dtype=np.float64)

            # Wrap lons to [0, 360) and sort
            lon360 = lon_src % 360.0
            sort_idx = np.argsort(lon360)
            lon_sorted = lon360[sort_idx]

            # S→N lat reversed to N→S for Atlas
            lat_desc = lat_src[::-1]

            # Target points
            target_lat = ATLAS_LAT.astype(np.float64)
            target_lon = ATLAS_LON.astype(np.float64)

            self.__class__._regrid_state = {
                "shape": (nlat_src, nlon_src),
                "sort_idx": sort_idx,
                "lat_desc": lat_desc,
                "lon_sorted": lon_sorted,
                "target_lat": target_lat,
                "target_lon": target_lon,
            }

        st = self._regrid_state

        # Reorder data: shift lons, flip lats
        data_shifted = data[:, st["sort_idx"]]
        data_flipped = data_shifted[::-1, :]

        # Interpolate using scipy (much faster than xarray for repeated calls)
        interp = RegularGridInterpolator(
            (st["lat_desc"], st["lon_sorted"]),
            data_flipped.astype(np.float64),
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate
        )

        # Build target mesh
        target_grid = np.stack(
            np.meshgrid(st["target_lat"], st["target_lon"], indexing="ij"),
            axis=-1,
        )
        result = interp(target_grid)
        return result.astype(np.float32)

    async def _download_asset(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
        local_path: pathlib.Path,
    ) -> None:
        """Download an asset to local cache."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with semaphore:
            temp_path = local_path.with_suffix(".tmp")
            try:
                async with client.stream(
                    "GET",
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                ) as response:
                    response.raise_for_status()
                    with temp_path.open("wb") as f:
                        async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                temp_path.replace(local_path)
            except Exception as error:
                temp_path.unlink(missing_ok=True)
                local_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download Met Office asset: {url}"
                ) from error

    def _local_path(self, href: str) -> pathlib.Path:
        """Generate a local cache path for a remote asset URL."""
        from urllib.parse import urlparse

        parsed = urlparse(href)
        suffix = pathlib.Path(parsed.path).suffix or ".nc"
        filename = hashlib.sha256(parsed.path.encode()).hexdigest() + suffix
        return pathlib.Path(self.cache) / filename

    @property
    def cache(self) -> str:
        """Return the cache directory."""
        cache_root = os.path.join(
            datasource_cache_root(), "planetary_computer", "met_office"
        )
        if not self._cache:
            cache_root = os.path.join(cache_root, "tmp")
        return cache_root
