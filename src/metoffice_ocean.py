"""Met Office Global Ocean SST data from AWS (ASDI).

This module provides :class:`MetOfficeASDI`, an earth2studio DataSource
that retrieves sea surface temperature from the Met Office's Global Coupled
Ocean model, published to the Amazon Sustainability Data Initiative (ASDI).

The data is on a regular 0.25° lat-lon grid (692×1440), updated daily.
SST is extracted as the surface (depth=0) layer of the 3-D potential
temperature field (``thetao``) and converted from °C to Kelvin.

Data source: https://registry.opendata.aws/met-office-global-ocean/

URL pattern::

    https://met-office-global-ocean-model-data.s3-eu-west-2.amazonaws.com/
    global-ocean-ORCA025/{YYYY}/{MM}/{DD}/T0000Z/
    level1_coupled_orca025_GL4_TEM_b{YYYYMMDD}_dm{YYYYMMDD}.nc

Usage::

    from src.metoffice_ocean import MetOfficeASDI

    ds = MetOfficeASDI()
    da = ds(["2026-02-17"], ["sst"])
    print(da.min().item(), da.max().item())  # ~271–310 K
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from datetime import datetime

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.data.utils import datasource_cache_root
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import httpx
except ImportError:
    httpx = None


#: S3 bucket base URL (eu-west-2, public, no auth required).
_S3_BASE = (
    "https://met-office-global-ocean-model-data.s3-eu-west-2.amazonaws.com"
)

#: URL template for the daily-mean TEM (potential temperature) T+0 analysis.
_URL_TEMPLATE = (
    "{base}/global-ocean-ORCA025/{year:04d}/{month:02d}/{day:02d}/T0000Z/"
    "level1_coupled_orca025_GL4_TEM_b{ymd}_dm{ymd}.nc"
)

#: Native grid coordinates (regular 0.25° lat-lon, already [0, 360]).
_LAT = np.linspace(-83.0, 89.75, 692, dtype=np.float32)
_LON = np.linspace(0.0, 359.75, 1440, dtype=np.float32)


def _build_url(date: datetime) -> str:
    ymd = f"{date.year:04d}{date.month:02d}{date.day:02d}"
    return _URL_TEMPLATE.format(
        base=_S3_BASE, year=date.year, month=date.month, day=date.day, ymd=ymd,
    )


def _url_to_cache_path(url: str, cache_dir: str) -> pathlib.Path:
    """Deterministic local path for a remote URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    filename = url.rsplit("/", 1)[-1]
    return pathlib.Path(cache_dir) / f"{url_hash}_{filename}"


class MetOfficeASDI:
    """Earth2Studio DataSource for Met Office Global Ocean SST from AWS ASDI.

    Fetches the daily-mean potential temperature analysis (T+0) and extracts
    the surface layer (depth=0) as SST.  Values are converted from °C to
    Kelvin to match the earth2studio convention.

    The data is on a regular 0.25° grid: 692 latitudes [-83, 89.75],
    1440 longitudes [0, 359.75].  NaN values over land are preserved;
    the framework's ``interp_to`` regridding handles them via nearest-neighbor
    interpolation.

    Parameters
    ----------
    cache : bool, optional
        Cache downloaded files locally, by default True.
    verbose : bool, optional
        Print progress information, by default True.
    request_timeout : int, optional
        HTTP request timeout in seconds, by default 120.
    """

    LAT_COORDS = _LAT
    LON_COORDS = _LON

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        request_timeout: int = 120,
    ) -> None:
        self._cache = cache
        self._verbose = verbose
        self._request_timeout = request_timeout

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch SST data.

        Parameters
        ----------
        time : datetime or list[datetime] or TimeArray
            Timestamps to fetch. Only the date portion is used (daily data).
        variable : str or list[str] or VariableArray
            Must be ``["sst"]``.

        Returns
        -------
        xr.DataArray
            SST in Kelvin, shape ``(time, variable, lat, lon)``.
        """
        if isinstance(time, datetime):
            times = [time]
        elif isinstance(time, np.ndarray):
            times = [t.astype("datetime64[ms]").astype(datetime) for t in time]
        else:
            times = list(time)

        if isinstance(variable, str):
            variables = [variable]
        elif isinstance(variable, np.ndarray):
            variables = list(variable)
        else:
            variables = list(variable)

        for v in variables:
            if v != "sst":
                raise ValueError(
                    f"MetOfficeASDI only provides 'sst', got '{v}'"
                )

        arrays = []
        for t in times:
            sst_k = self._fetch_sst_kelvin(t)
            arrays.append(sst_k)

        data = np.stack(arrays, axis=0)  # (time, lat, lon)
        data = data[:, np.newaxis, :, :]  # (time, variable=1, lat, lon)

        da = xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": np.array(
                    [np.datetime64(t, "ns") for t in times],
                    dtype="datetime64[ns]",
                ),
                "variable": np.array(variables),
                "lat": self.LAT_COORDS,
                "lon": self.LON_COORDS,
            },
        )
        return da

    def _fetch_sst_kelvin(self, time: datetime) -> np.ndarray:
        """Download (or load from cache) and return SST in Kelvin."""
        local_path = self._download(time)
        with xr.open_dataset(local_path, engine="h5netcdf") as ds:
            # depth=0 is the ocean surface
            sst_c = (
                ds["thetao"]
                .isel(time=0)
                .sel(depth=0.0)
                .values
                .astype(np.float32)
            )
        return sst_c + np.float32(273.15)

    def _download(self, time: datetime) -> pathlib.Path:
        """Download a TEM file if not already cached, return local path."""
        if httpx is None:
            raise ImportError("httpx is required: pip install httpx")

        url = _build_url(time)
        local_path = _url_to_cache_path(url, self._cache_dir)

        if local_path.exists():
            if self._verbose:
                logger.debug(f"MetOfficeASDI: cache hit {local_path.name}")
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(".tmp")

        if self._verbose:
            logger.info(f"MetOfficeASDI: downloading {url}")

        with httpx.stream(
            "GET", url, timeout=self._request_timeout, follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)

        tmp_path.rename(local_path)

        if self._verbose:
            size_mb = local_path.stat().st_size / 1e6
            logger.info(f"MetOfficeASDI: saved {local_path.name} ({size_mb:.1f} MB)")

        return local_path

    @property
    def _cache_dir(self) -> str:
        root = os.path.join(datasource_cache_root(), "metoffice_asdi_sst")
        if not self._cache:
            root = os.path.join(root, "tmp")
        return root
