"""Tests for the Met Office → Atlas regridding pipeline.

Verifies that:
1. The longitude periodicity padding produces no NaN after interpolation.
2. The full atmospheric → diagnostic pipeline has no missing data on the
   Atlas 0.25° grid (721×1440).
3. The ocean SST source grid already covers lon=0 (no padding needed).

All tests use synthetic data on the real coordinate grids — no network
access required.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import torch
import xarray as xr

from src.metoffice_atmospheric import (
    MetOfficePlanetaryComputer,
    _pad_lon_periodic,
)
from src.metoffice_diagnostic import (
    INPUT_VARIABLES,
    METOFFICE_VARIABLES,
    OUTPUT_VARIABLES,
    MetOfficeAtlasDiagnostic,
)
from src.metoffice_ocean import MetOfficeASDI
from src.metoffice_source import MetOfficeAtlasSource, _ATLAS_LAT, _ATLAS_LON


# ---- Coordinate constants ----

_NATIVE_NLAT = 1920
_NATIVE_NLON = 2560
_RAW_LON = np.linspace(-179.9296875, 179.9296875, _NATIVE_NLON, dtype=np.float32)
_NATIVE_LON = np.sort(_RAW_LON % 360)
_NATIVE_LAT = np.linspace(-89.953125, 89.953125, _NATIVE_NLAT, dtype=np.float32)


def _make_native_da(
    n_vars: int = 1,
    fill: float = 1.0,
) -> xr.DataArray:
    """Create a synthetic DataArray on the native Met Office grid."""
    times = np.array([np.datetime64("2026-01-24T00:00", "ns")])
    variables = np.array([f"var{i}" for i in range(n_vars)])
    data = np.full(
        (1, n_vars, _NATIVE_NLAT, _NATIVE_NLON), fill, dtype=np.float32,
    )
    return xr.DataArray(
        data,
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": times,
            "variable": variables,
            "lat": _NATIVE_LAT,
            "lon": _NATIVE_LON,
        },
    )


# ---- Tests for _pad_lon_periodic ----


class TestPadLonPeriodic:
    """Tests for the longitude periodicity padding function."""

    def test_adds_two_columns(self):
        da = _make_native_da()
        padded = _pad_lon_periodic(da)
        assert padded.sizes["lon"] == _NATIVE_NLON + 2

    def test_prepended_column_is_below_zero(self):
        da = _make_native_da()
        padded = _pad_lon_periodic(da)
        assert padded.lon.values[0] < 0.0

    def test_appended_column_is_above_360(self):
        da = _make_native_da()
        padded = _pad_lon_periodic(da)
        assert padded.lon.values[-1] > 360.0

    def test_prepended_values_match_last_column(self):
        """The wrap-around column at lon<0 should hold data from the last native column."""
        da = _make_native_da()
        # Set the last column to a distinctive value
        da.values[:, :, :, -1] = 42.0
        padded = _pad_lon_periodic(da)
        np.testing.assert_array_equal(padded.isel(lon=0).values, 42.0)

    def test_appended_values_match_first_column(self):
        """The wrap-around column at lon>360 should hold data from the first native column."""
        da = _make_native_da()
        da.values[:, :, :, 0] = 99.0
        padded = _pad_lon_periodic(da)
        np.testing.assert_array_equal(padded.isel(lon=-1).values, 99.0)

    def test_interior_unchanged(self):
        da = _make_native_da(fill=7.0)
        padded = _pad_lon_periodic(da)
        interior = padded.isel(lon=slice(1, -1))
        np.testing.assert_array_equal(interior.values, 7.0)

    def test_padded_range_covers_atlas_grid(self):
        """After padding, the lon range must contain every Atlas grid point."""
        da = _make_native_da()
        padded = _pad_lon_periodic(da)
        assert padded.lon.values[0] < _ATLAS_LON[0]
        assert padded.lon.values[-1] > _ATLAS_LON[-1]


# ---- Tests for interpolation to the Atlas grid ----


class TestInterpolationNoNaN:
    """Verify that interpolating padded native data to the Atlas grid produces no NaN."""

    def test_interp_to_atlas_lon_no_nan(self):
        """Bilinear interpolation from padded native → Atlas grid must have no NaN."""
        da = _make_native_da(fill=300.0)
        padded = _pad_lon_periodic(da)

        target_lat = xr.DataArray(_ATLAS_LAT, dims=["_lat"])
        target_lon = xr.DataArray(_ATLAS_LON, dims=["_lon"])

        result = padded.interp(lat=target_lat, lon=target_lon, method="linear")
        assert not np.any(np.isnan(result.values)), (
            f"Found {np.isnan(result.values).sum()} NaN values after interpolation"
        )

    def test_interp_nearest_no_nan(self):
        """Nearest-neighbor interp (the fetch_data default) must also have no NaN."""
        da = _make_native_da(fill=300.0)
        padded = _pad_lon_periodic(da)

        target_lat = xr.DataArray(_ATLAS_LAT, dims=["_lat"])
        target_lon = xr.DataArray(_ATLAS_LON, dims=["_lon"])

        result = padded.interp(lat=target_lat, lon=target_lon, method="nearest")
        assert not np.any(np.isnan(result.values)), (
            f"Found {np.isnan(result.values).sum()} NaN values after nearest interp"
        )

    def test_without_padding_has_nan_at_lon_zero(self):
        """Confirm the bug: without padding, lon=0 produces NaN."""
        da = _make_native_da(fill=300.0)
        # No padding — use the raw native grid
        target_lat = xr.DataArray(_ATLAS_LAT, dims=["_lat"])
        target_lon = xr.DataArray(_ATLAS_LON, dims=["_lon"])

        result = da.interp(lat=target_lat, lon=target_lon, method="linear")
        # lon=0 should be NaN because it's below the native grid minimum (0.0703)
        lon0_slice = result.sel(_lon=0.0)
        assert np.all(np.isnan(lon0_slice.values)), (
            "Expected NaN at lon=0 without padding (this confirms the bug exists)"
        )

    def test_interp_preserves_smooth_field(self):
        """A smooth sinusoidal field should survive regridding without large errors."""
        da = _make_native_da()
        lon_rad = np.deg2rad(_NATIVE_LON)[np.newaxis, np.newaxis, np.newaxis, :]
        lat_rad = np.deg2rad(_NATIVE_LAT)[np.newaxis, np.newaxis, :, np.newaxis]
        da.values[:] = np.cos(lat_rad) * np.sin(lon_rad)

        padded = _pad_lon_periodic(da)
        target_lat = xr.DataArray(_ATLAS_LAT, dims=["_lat"])
        target_lon = xr.DataArray(_ATLAS_LON, dims=["_lon"])
        result = padded.interp(lat=target_lat, lon=target_lon, method="linear")

        assert not np.any(np.isnan(result.values))
        # At lon=0, sin(0)=0, so the field should be near zero
        lon0_vals = result.sel(_lon=0.0).values
        assert np.abs(lon0_vals).max() < 0.01, (
            f"Expected near-zero at lon=0 for sin(lon) field, got max={np.abs(lon0_vals).max():.4f}"
        )


# ---- Tests for ocean SST grid coverage ----


class TestOceanSSTGrid:
    """Verify the ocean SST source grid needs no padding."""

    def test_ocean_lon_starts_at_zero(self):
        assert MetOfficeASDI.LON_COORDS[0] == 0.0

    def test_ocean_lon_covers_atlas(self):
        """Ocean grid [0, 359.75] matches the Atlas grid exactly."""
        np.testing.assert_array_equal(MetOfficeASDI.LON_COORDS, _ATLAS_LON.astype(np.float32))


# ---- Tests for diagnostic (no NaN propagation) ----


class TestDiagnosticNoNaN:
    """Run the diagnostic on synthetic data and verify no NaN in output."""

    def test_diagnostic_no_nan(self):
        diag = MetOfficeAtlasDiagnostic()
        n_in = len(INPUT_VARIABLES)
        nlat, nlon = 4, 8  # small grid for speed

        x = torch.ones(1, n_in, nlat, nlon, dtype=torch.float32)
        # Set plausible values so derived quantities don't blow up
        for i, var in enumerate(INPUT_VARIABLES):
            if "air_temperature" in var or var == "air_temperature_at_screen_level":
                x[:, i] = 280.0  # ~7°C
            elif "air_pressure" in var:
                x[:, i] = 101325.0  # Pa
            elif "wind_speed" in var:
                x[:, i] = 5.0  # m/s
            elif "wind_from_direction" in var:
                x[:, i] = 225.0  # degrees (SW)
            elif "relative_humidity" in var:
                x[:, i] = 0.7  # 70% (fractional)
            elif "geopotential_height" in var:
                x[:, i] = 5000.0  # m
            elif var == "sst":
                x[:, i] = 290.0  # K
            elif var == "lwe_precipitation_rate":
                x[:, i] = 0.001  # kg/m²/s

        coords = OrderedDict(
            batch=np.empty(0),
            variable=np.array(INPUT_VARIABLES),
            lat=np.linspace(90, -90, nlat),
            lon=np.linspace(0, 359, nlon),
        )
        out, out_coords = diag(x, coords)
        assert out.shape == (1, len(OUTPUT_VARIABLES), nlat, nlon)
        nan_count = torch.isnan(out).sum().item()
        assert nan_count == 0, f"Diagnostic produced {nan_count} NaN values"


# ---- Integration: full pipeline with mocked data sources ----


class TestFullPipelineNoNaN:
    """End-to-end test with mocked data sources on the real grids."""

    def test_atlas_source_no_nan(self):
        """MetOfficeAtlasSource should produce no NaN on the Atlas grid."""
        source = MetOfficeAtlasSource()

        # Create synthetic atmospheric data on the padded native grid
        n_atmos = len(METOFFICE_VARIABLES)
        times = np.array([np.datetime64("2026-01-24T00:00", "ns")])

        atmos_da = xr.DataArray(
            np.full((1, n_atmos, _NATIVE_NLAT, _NATIVE_NLON), 1.0, dtype=np.float32),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": times,
                "variable": np.array(METOFFICE_VARIABLES),
                "lat": _NATIVE_LAT,
                "lon": _NATIVE_LON,
            },
        )
        # Set physically plausible values
        for i, var in enumerate(METOFFICE_VARIABLES):
            if "air_temperature" in var:
                atmos_da.values[:, i] = 280.0
            elif "air_pressure" in var:
                atmos_da.values[:, i] = 101325.0
            elif "wind_speed" in var:
                atmos_da.values[:, i] = 5.0
            elif "wind_from_direction" in var:
                atmos_da.values[:, i] = 225.0
            elif "relative_humidity" in var:
                atmos_da.values[:, i] = 0.7
            elif "geopotential_height" in var:
                atmos_da.values[:, i] = 5000.0
            elif var == "lwe_precipitation_rate":
                atmos_da.values[:, i] = 0.001

        # Apply the periodicity padding (as the real fetch() now does)
        atmos_da_padded = _pad_lon_periodic(atmos_da)

        # Create synthetic SST on the ocean grid (already aligned)
        ocean_da = xr.DataArray(
            np.full((1, 1, 692, 1440), 290.0, dtype=np.float32),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": times,
                "variable": np.array(["sst"]),
                "lat": MetOfficeASDI.LAT_COORDS,
                "lon": MetOfficeASDI.LON_COORDS,
            },
        )

        # Mock both data sources to return our synthetic data
        def mock_atmos_call(time, variable):
            var_list = list(variable) if not isinstance(variable, str) else [variable]
            var_indices = [list(METOFFICE_VARIABLES).index(v) for v in var_list]
            return atmos_da_padded.sel(variable=var_list)

        def mock_ocean_call(time, variable):
            return ocean_da

        source.atmos_ds = type("MockAtmos", (), {"__call__": lambda self, t, v: mock_atmos_call(t, v)})()
        source.ocean_ds = type("MockOcean", (), {"__call__": lambda self, t, v: mock_ocean_call(t, v)})()

        result = source(
            time=datetime(2026, 1, 24),
            variable=OUTPUT_VARIABLES,
        )

        assert result.shape == (1, len(OUTPUT_VARIABLES), 721, 1440)

        nan_mask = np.isnan(result.values)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            # Report which variables and where
            for vi, var in enumerate(OUTPUT_VARIABLES):
                var_nans = np.isnan(result.values[0, vi])
                if var_nans.any():
                    rows, cols = np.where(var_nans)
                    lons_with_nan = sorted(set(_ATLAS_LON[cols]))[:5]
                    pytest.fail(
                        f"Variable '{var}' has {var_nans.sum()} NaN values. "
                        f"Sample lon values with NaN: {lons_with_nan}"
                    )

    def test_lon_zero_column_not_nan(self):
        """Specifically verify the lon=0 column has no NaN after regridding."""
        da = _make_native_da(n_vars=3, fill=300.0)
        padded = _pad_lon_periodic(da)

        target_lat = xr.DataArray(_ATLAS_LAT, dims=["_lat"])
        target_lon = xr.DataArray(_ATLAS_LON, dims=["_lon"])

        result = padded.interp(lat=target_lat, lon=target_lon, method="nearest")
        lon0 = result.sel(_lon=0.0)
        assert not np.any(np.isnan(lon0.values)), "lon=0 column still contains NaN"

    def test_all_atlas_lon_columns_populated(self):
        """Every longitude column in the Atlas grid must have data."""
        da = _make_native_da(fill=1.0)
        padded = _pad_lon_periodic(da)

        target_lat = xr.DataArray(_ATLAS_LAT, dims=["_lat"])
        target_lon = xr.DataArray(_ATLAS_LON, dims=["_lon"])

        result = padded.interp(lat=target_lat, lon=target_lon, method="linear")
        for i, lon_val in enumerate(_ATLAS_LON):
            col = result.isel(_lon=i).values
            assert not np.any(np.isnan(col)), (
                f"Column at lon={lon_val:.2f}° (index {i}) contains NaN"
            )
