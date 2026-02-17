# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic model that transforms raw Met Office fields into Atlas model inputs.

This module provides :class:`MetOfficeToAtlasDiagnostic`, a
``DiagnosticModel``-protocol ``torch.nn.Module`` that converts native
Met Office variables (wind speed/direction, relative humidity, geopotential
height, etc.) into the derived variables that Atlas expects (u/v wind
components, specific humidity, geopotential, etc.).

The diagnostic operates on torch tensors with coordinate system:
``(batch, variable, lat, lon)``.

The diagnostic expects SST as an input (from OISST or another source)
alongside the native Met Office fields.  Use :func:`fetch_oisst` to
retrieve SST and :func:`combine_inputs` to merge it with Met Office data
before calling the diagnostic.

Usage::

    from metoffice_diagnostic import MetOfficeToAtlasDiagnostic, fetch_oisst, combine_inputs

    diag = MetOfficeToAtlasDiagnostic()
    # metoffice_variables = diag.metoffice_variables  (70 native Met Office vars)
    # fetch Met Office and OISST separately, then combine:
    x_combined, coords_combined = combine_inputs(x_metoffice, coords_metoffice,
                                                  sst_data, sst_coords)
    x_atlas, coords_atlas = diag(x_combined, coords_combined)
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem, TimeArray

#: Standard gravity (m s⁻²).
G = 9.80665

#: Ratio of molecular weight of water vapor to dry air.
EPSILON = 0.622

#: Pressure levels served by the Atlas model (hPa).
ATLAS_PRESSURE_LEVELS_HPA: list[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]


def _build_metoffice_variables() -> list[str]:
    """Native Met Office variables needed by this diagnostic."""
    variables: list[str] = [
        "wind_speed_at_10m",
        "wind_from_direction_at_10m",
        "air_temperature_at_screen_level",
        "air_pressure_at_sea_level",
        "lwe_precipitation_rate",
    ]
    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        variables.extend([
            f"wind_speed_{hpa}hPa",
            f"wind_from_direction_{hpa}hPa",
            f"air_temperature_{hpa}hPa",
            f"relative_humidity_{hpa}hPa",
            f"geopotential_height_{hpa}hPa",
        ])
    return variables


#: Native Met Office variables (70 fields).
METOFFICE_VARIABLES: list[str] = _build_metoffice_variables()

#: Full diagnostic input: Met Office fields + SST from OISST.
INPUT_VARIABLES: list[str] = METOFFICE_VARIABLES + ["sst"]

#: Atlas output variables (75 fields).
OUTPUT_VARIABLES: list[str] = (
    ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv"]
    + [f"{p}{hpa}" for p in ("u", "v", "z", "t", "q") for hpa in ATLAS_PRESSURE_LEVELS_HPA]
    + ["sst", "tp"]
)


class MetOfficeToAtlasDiagnostic(torch.nn.Module):
    """Diagnostic model converting native Met Office fields + SST to Atlas inputs.

    Derivations performed:

    - **Wind components**: ``u = -speed × sin(dir)``,
      ``v = -speed × cos(dir)`` for 10 m and all pressure levels.
      Meteorological convention: direction is "from", clockwise from north.
    - **Specific humidity**: from relative humidity + temperature + pressure
      via Bolton (1980) saturation vapor pressure.
    - **Geopotential**: ``z = height × 9.80665``.
    - **100 m wind**: falls back to 10 m (not available from Met Office).
    - **Surface pressure**: approximated by MSLP.
    - **TCWV**: filled with zeros (not available from Met Office).
    - **SST**: passed through from input (supplied by OISST or similar).

    The module is stateless and operates in inference mode.
    """

    def __init__(self) -> None:
        super().__init__()

        self.in_variables = np.array(INPUT_VARIABLES)
        self.out_variables = np.array(OUTPUT_VARIABLES)
        #: The 70 Met Office variable names (for fetching from the native source).
        self.metoffice_variables = np.array(METOFFICE_VARIABLES)

        self._in_idx: dict[str, int] = {
            v: i for i, v in enumerate(INPUT_VARIABLES)
        }
        self._out_idx: dict[str, int] = {
            v: i for i, v in enumerate(OUTPUT_VARIABLES)
        }

        p_pa = torch.tensor(
            [float(hpa * 100) for hpa in ATLAS_PRESSURE_LEVELS_HPA],
            dtype=torch.float32,
        )
        self.register_buffer("_pressure_pa", p_pa)

    def input_coords(self) -> CoordSystem:
        """Coordinate system expected by this diagnostic."""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": self.in_variables.copy(),
                "lat": np.empty(0),
                "lon": np.empty(0),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Coordinate system produced by this diagnostic."""
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = self.out_variables.copy()
        return output_coords

    def _in(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x[:, self._in_idx[name]]

    @staticmethod
    def _wind_components(
        speed: torch.Tensor, direction_deg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Meteorological wind speed + direction → (u, v).

        Direction is "from", clockwise from north.
        u = -speed × sin(dir), v = -speed × cos(dir).
        """
        direction_rad = direction_deg * (math.pi / 180.0)
        u = -speed * torch.sin(direction_rad)
        v = -speed * torch.cos(direction_rad)
        return u, v

    @staticmethod
    def _saturation_vapor_pressure(t_k: torch.Tensor) -> torch.Tensor:
        """Saturation vapor pressure (Pa) from temperature (K) via Bolton (1980)."""
        t_c = t_k - 273.15
        return 611.2 * torch.exp(17.67 * t_c / (t_c + 243.5))

    @classmethod
    def _specific_humidity(
        cls,
        rh_frac: torch.Tensor,
        t_k: torch.Tensor,
        p_pa: torch.Tensor,
    ) -> torch.Tensor:
        """Specific humidity (kg kg⁻¹) from fractional RH, T (K), P (Pa)."""
        e_s = cls._saturation_vapor_pressure(t_k)
        e = rh_frac * e_s
        q = EPSILON * e / (p_pa - (1.0 - EPSILON) * e)
        return torch.clamp(q, min=0.0)

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Transform native Met Office fields + SST to Atlas inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, variable, lat, lon)``.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system with Atlas variables.
        """
        output_coords = self.output_coords(coords)
        batch_size = x.shape[0]
        nlat = x.shape[2]
        nlon = x.shape[3]
        n_out = len(self.out_variables)

        out = torch.zeros(
            (batch_size, n_out, nlat, nlon),
            dtype=x.dtype,
            device=x.device,
        )

        ws10 = self._in(x, "wind_speed_at_10m")
        wd10 = self._in(x, "wind_from_direction_at_10m")
        u10, v10 = self._wind_components(ws10, wd10)
        out[:, self._out_idx["u10m"]] = u10
        out[:, self._out_idx["v10m"]] = v10

        # 100 m wind not available from Met Office; fall back to 10 m
        out[:, self._out_idx["u100m"]] = u10
        out[:, self._out_idx["v100m"]] = v10

        out[:, self._out_idx["t2m"]] = self._in(x, "air_temperature_at_screen_level")

        # Surface pressure not available; approximate with MSLP
        msl = self._in(x, "air_pressure_at_sea_level")
        out[:, self._out_idx["sp"]] = msl
        out[:, self._out_idx["msl"]] = msl

        # TCWV not available; left as zero from initialization

        for level_i, hpa in enumerate(ATLAS_PRESSURE_LEVELS_HPA):
            p_pa = self._pressure_pa[level_i]

            ws = self._in(x, f"wind_speed_{hpa}hPa")
            wd = self._in(x, f"wind_from_direction_{hpa}hPa")
            u, v = self._wind_components(ws, wd)
            out[:, self._out_idx[f"u{hpa}"]] = u
            out[:, self._out_idx[f"v{hpa}"]] = v

            out[:, self._out_idx[f"t{hpa}"]] = self._in(
                x, f"air_temperature_{hpa}hPa"
            )

            out[:, self._out_idx[f"z{hpa}"]] = (
                self._in(x, f"geopotential_height_{hpa}hPa") * G
            )

            rh = self._in(x, f"relative_humidity_{hpa}hPa")
            t_k = self._in(x, f"air_temperature_{hpa}hPa")
            q = self._specific_humidity(rh, t_k, p_pa)
            out[:, self._out_idx[f"q{hpa}"]] = q

        out[:, self._out_idx["sst"]] = self._in(x, "sst")
        out[:, self._out_idx["tp"]] = self._in(x, "lwe_precipitation_rate")

        return out, output_coords


# ---- Data loading helpers ----


def combine_inputs(
    metoffice_data: torch.Tensor,
    metoffice_coords: CoordSystem,
    sst_data: torch.Tensor,
    sst_coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Combine Met Office fields and OISST into a single input tensor.

    Concatenates along the variable dimension and builds a coordinate
    system matching :data:`INPUT_VARIABLES`.

    Parameters
    ----------
    metoffice_data : torch.Tensor
        Met Office fields, shape ``(batch, 70, lat, lon)``.
    metoffice_coords : CoordSystem
        Coordinates from the Met Office fetch.
    sst_data : torch.Tensor
        SST field from OISST, shape ``(batch, 1, lat, lon)``.
    sst_coords : CoordSystem
        Coordinates from the OISST fetch.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Combined tensor ``(batch, 71, lat, lon)`` and coordinate system.
    """
    combined = torch.cat([metoffice_data, sst_data], dim=1)
    coords = metoffice_coords.copy()
    coords["variable"] = np.array(INPUT_VARIABLES)
    return combined, coords


def fetch_oisst(
    time: TimeArray,
    *,
    atlas_input_coords: CoordSystem,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, CoordSystem]:
    """Fetch SST from NOAA OISST on Planetary Computer, regridded to the Atlas grid.

    Parameters
    ----------
    time : TimeArray
        Times to fetch (same as passed to the Met Office data source).
    atlas_input_coords : CoordSystem
        Atlas model input coords (used for regridding via ``interp_to``).
    device : torch.device or str
        Target device for the output tensor.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        SST field ``(batch, 1, lat, lon)`` in Kelvin, and its coordinates.
    """
    from earth2studio.data import PlanetaryComputerOISST
    from earth2studio.data.utils import fetch_data

    oisst = PlanetaryComputerOISST()
    return fetch_data(
        source=oisst,
        time=time,
        variable=np.array(["sst"]),
        device=device,
        interp_to=atlas_input_coords,
    )
