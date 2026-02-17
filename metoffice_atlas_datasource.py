# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""Composite DataSource that returns Atlas-ready fields from Met Office data.

:class:`MetOfficeAtlasDataSource` wraps the full Met Office → SST → diagnostic
pipeline into a single earth2studio :class:`~earth2studio.data.base.DataSource`.
When called with Atlas variable names and a time, it internally:

1. Fetches native Met Office fields and regrids to the Atlas 0.25° grid.
2. Fetches SST from the Met Office Global Ocean analysis.
3. Combines the two and runs :class:`MetOfficeToAtlasDiagnostic` to derive
   all 75 Atlas variables.
4. Returns an ``xr.DataArray`` with dims ``(time, variable, lat, lon)``.

This makes it directly usable with :func:`earth2studio.run.deterministic`::

    from earth2studio.run import deterministic
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px.atlas import Atlas
    from metoffice_atlas_datasource import MetOfficeAtlasDataSource

    model = Atlas.load_model(Atlas.load_default_package())
    io = deterministic(
        time=["2026-02-17T00:00"],
        nsteps=4,
        prognostic=model,
        data=MetOfficeAtlasDataSource(),
        io=ZarrBackend(),
    )
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from earth2studio.data.utils import fetch_data
from earth2studio.utils.type import TimeArray, VariableArray

from coords import interp_coords_to_latlon, make_interp_to
from metoffice_diagnostic import (
    MetOfficeToAtlasDiagnostic,
    OUTPUT_VARIABLES,
    combine_inputs,
)
from metoffice_native import PlanetaryComputerMetOfficeNative
from metoffice_ocean_sst import MetOfficeOceanSST


#: Atlas 0.25° grid coordinates.
_ATLAS_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float64)
_ATLAS_LON = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float64)


def _atlas_interp_to() -> OrderedDict:
    """Build the ``interp_to`` dict for regridding to the Atlas grid."""
    return OrderedDict({"_lat": _ATLAS_LAT, "_lon": _ATLAS_LON})


class MetOfficeAtlasDataSource:
    """Earth2Studio DataSource that produces Atlas-ready fields from Met Office data.

    Each call runs the full pipeline: native Met Office fetch → SST fetch →
    regrid to 0.25° → diagnostic derivation → 75 Atlas variables.  The result
    is on the Atlas grid (721×1440, lat 90→-90, lon 0→359.75).

    Parameters
    ----------
    forecast_hour : int, optional
        Met Office forecast lead time to use, by default 0 (T+0 analysis).
    cache : bool, optional
        Cache downloaded data files locally, by default True.
    verbose : bool, optional
        Print progress information, by default True.
    """

    def __init__(
        self,
        forecast_hour: int = 0,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        self.native_ds = PlanetaryComputerMetOfficeNative(
            forecast_hour=forecast_hour,
            cache=cache,
            verbose=verbose,
        )
        self.sst_ds = MetOfficeOceanSST(cache=cache, verbose=verbose)
        self.diagnostic = MetOfficeToAtlasDiagnostic()
        self._interp_to = _atlas_interp_to()
        self._all_variables = np.array(OUTPUT_VARIABLES)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch Atlas-ready fields for the given times and variables.

        Parameters
        ----------
        time : datetime or list[datetime] or TimeArray
            Timestamps to fetch. Must fall on 6-hourly Met Office run
            boundaries (00, 06, 12, 18 UTC).
        variable : str or list[str] or VariableArray
            Atlas variable names to return (subset of the 75 output variables).

        Returns
        -------
        xr.DataArray
            Data with dims ``(time, variable, lat, lon)`` on the Atlas grid.
        """
        if isinstance(time, datetime):
            times = np.array([np.datetime64(time, "ns")])
        elif isinstance(time, list):
            times = np.array([np.datetime64(t, "ns") for t in time])
        else:
            times = time

        if isinstance(variable, str):
            variables = np.array([variable])
        elif isinstance(variable, list):
            variables = np.array(variable)
        else:
            variables = np.array(variable)

        # Run full pipeline for all variables, then subset at the end
        atlas_data = self._fetch_atlas_fields(times)

        # Subset to requested variables
        all_vars = list(self._all_variables)
        var_indices = [all_vars.index(v) for v in variables]
        subset = atlas_data[:, var_indices, :, :]

        da = xr.DataArray(
            subset,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": times.astype("datetime64[ns]"),
                "variable": variables,
                "lat": _ATLAS_LAT,
                "lon": _ATLAS_LON,
            },
        )
        return da

    def _fetch_atlas_fields(self, times: np.ndarray) -> np.ndarray:
        """Run the full pipeline and return all 75 Atlas variables.

        Returns numpy array of shape ``(len(times), 75, 721, 1440)``.
        """
        metoffice_vars = self.diagnostic.metoffice_variables

        # Fetch Met Office native data, regridded to Atlas grid
        x_mo, coords_mo = fetch_data(
            source=self.native_ds,
            time=times,
            variable=metoffice_vars,
            device=torch.device("cpu"),
            interp_to=self._interp_to,
        )
        coords_mo = interp_coords_to_latlon(coords_mo)

        # Fetch SST, regridded to Atlas grid
        atlas_coords = OrderedDict({"lat": _ATLAS_LAT, "lon": _ATLAS_LON})
        x_sst, coords_sst = fetch_data(
            source=self.sst_ds,
            time=times,
            variable=np.array(["sst"]),
            device=torch.device("cpu"),
            interp_to=self._interp_to,
        )
        coords_sst = interp_coords_to_latlon(coords_sst)

        # Combine: (time, lead_time=1, 71, lat, lon)
        x_combined, coords_combined = combine_inputs(
            x_mo, coords_mo, x_sst, coords_sst,
        )

        # Run diagnostic: derives all 75 Atlas variables
        x_atlas, _coords_atlas = self.diagnostic(x_combined, coords_combined)

        # x_atlas shape: (time, lead_time=1, 75, 721, 1440)
        # Squeeze lead_time and convert to numpy
        return x_atlas.squeeze(1).numpy()
