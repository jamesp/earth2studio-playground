"""Composite DataSource returning Atlas-ready fields from Met Office data.

:class:`MetOfficeAtlasSource` wraps the full Met Office → SST → diagnostic
pipeline into a single earth2studio DataSource.  When called with Atlas
variable names and a time, it internally:

1. Fetches native Met Office atmospheric fields and regrids to the Atlas 0.25° grid.
2. Fetches SST from the Met Office Global Ocean analysis and regrids to the Atlas grid.
3. Combines the two (70 atmos + 1 SST) and runs :class:`MetOfficeAtlasDiagnostic`
   to derive all 75 Atlas variables.
4. Returns an ``xr.DataArray`` with dims ``(time, variable, lat, lon)``.

Regridding uses xarray's ``DataArray.interp()`` (bilinear), triggered by the
``interp_to`` argument of :func:`~earth2studio.data.utils.fetch_data`.  The
atmospheric source normalizes lon to [0, 360] before returning the DataArray,
so the interpolation target and source share the same convention throughout.

This makes it directly usable with :func:`earth2studio.run.deterministic`::

    from earth2studio.run import deterministic
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px.atlas import Atlas
    from src.metoffice_source import MetOfficeAtlasSource

    model = Atlas.load_model(Atlas.load_default_package())
    io = deterministic(
        time=["2026-02-17T00:00"],
        nsteps=4,
        prognostic=model,
        data=MetOfficeAtlasSource(),
        io=ZarrBackend("forecast.zarr"),
    )
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from earth2studio.data.utils import fetch_data
from earth2studio.utils.type import CoordSystem, TimeArray, VariableArray

from src.metoffice_atmospheric import MetOfficePlanetaryComputer
from src.metoffice_diagnostic import MetOfficeAtlasDiagnostic, OUTPUT_VARIABLES
from src.metoffice_ocean import MetOfficeASDI


#: Atlas 0.25° grid: lat N→S (90→-90), lon [0, 359.75].
_ATLAS_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float64)
_ATLAS_LON = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float64)


def _interp_coords_to_latlon(coords: CoordSystem) -> CoordSystem:
    """Rename ``_lat``/``_lon`` → ``lat``/``lon`` in a coordinate system.

    ``fetch_data`` with ``interp_to`` produces ``_lat``/``_lon`` keys in the
    output coords; models expect ``lat``/``lon``.
    """
    out = OrderedDict()
    for key, val in coords.items():
        if key == "_lat":
            out["lat"] = val
        elif key == "_lon":
            out["lon"] = val
        else:
            out[key] = val
    return out


class MetOfficeAtlasSource:
    """Earth2Studio DataSource that produces Atlas-ready fields from Met Office data.

    Each call runs the full pipeline: native Met Office fetch → SST fetch →
    regrid to 0.25° → diagnostic derivation → 75 Atlas variables.  The result
    is on the Atlas grid (721×1440, lat 90→-90, lon 0→359.75).

    Parameters
    ----------
    forecast_hour : int, optional
        Met Office atmospheric forecast lead time, by default 0 (T+0 analysis).
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
        self.atmos_ds = MetOfficePlanetaryComputer(
            forecast_hour=forecast_hour,
            cache=cache,
            verbose=verbose,
        )
        self.ocean_ds = MetOfficeASDI(cache=cache, verbose=verbose)
        self.diagnostic = MetOfficeAtlasDiagnostic()

        # interp_to uses _lat/_lon keys as required by fetch_data
        self._interp_to = OrderedDict({"_lat": _ATLAS_LAT, "_lon": _ATLAS_LON})
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
        """Run the full pipeline and return all 75 Atlas variables as numpy.

        Returns array of shape ``(len(times), 75, 721, 1440)``.
        """
        metoffice_vars = self.diagnostic.metoffice_variables

        # Step 1: fetch atmospheric data on native ~0.09° grid, regrid to Atlas
        x_mo, coords_mo = fetch_data(
            source=self.atmos_ds,
            time=times,
            variable=metoffice_vars,
            device=torch.device("cpu"),
            interp_to=self._interp_to,
        )
        coords_mo = _interp_coords_to_latlon(coords_mo)

        # Step 2: fetch SST on native 0.25° grid, regrid/align to Atlas
        x_sst, coords_sst = fetch_data(
            source=self.ocean_ds,
            time=times,
            variable=np.array(["sst"]),
            device=torch.device("cpu"),
            interp_to=self._interp_to,
        )
        coords_sst = _interp_coords_to_latlon(coords_sst)

        # Step 3: combine (time, lead_time, 70, lat, lon) + (time, lead_time, 1, lat, lon)
        var_dim = list(coords_mo.keys()).index("variable")
        x_combined = torch.cat([x_mo, x_sst], dim=var_dim)
        coords_combined = coords_mo.copy()
        from src.metoffice_diagnostic import INPUT_VARIABLES
        coords_combined["variable"] = np.array(INPUT_VARIABLES)

        # Step 4: run diagnostic — pure variable derivation, no regridding
        x_atlas, _coords_atlas = self.diagnostic(x_combined, coords_combined)

        # x_atlas shape: (time, lead_time=1, 75, 721, 1440)
        return x_atlas.squeeze(1).numpy()
