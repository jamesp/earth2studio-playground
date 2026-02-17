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

Usage::

    from metoffice_diagnostic import MetOfficeToAtlasDiagnostic

    diag = MetOfficeToAtlasDiagnostic()
    # input_coords() tells you what native Met Office variables are needed
    # output_coords() tells you what Atlas variables are produced
    x_atlas, coords_atlas = diag(x_metoffice, coords_metoffice)
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

#: Standard gravity (m s⁻²).
G = 9.80665

#: Ratio of molecular weight of water vapour to dry air.
EPSILON = 0.622

#: Pressure levels served by the Atlas model (hPa).
ATLAS_PRESSURE_LEVELS_HPA: list[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]


def _build_input_variables() -> list[str]:
    """Native Met Office variables needed by this diagnostic."""
    variables: list[str] = []

    variables.append("wind_speed_at_10m")
    variables.append("wind_from_direction_at_10m")
    variables.append("air_temperature_at_screen_level")
    variables.append("air_pressure_at_sea_level")
    variables.append("surface_temperature")
    variables.append("lwe_precipitation_rate")

    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        variables.append(f"wind_speed_{hpa}hPa")
        variables.append(f"wind_from_direction_{hpa}hPa")
        variables.append(f"air_temperature_{hpa}hPa")
        variables.append(f"relative_humidity_{hpa}hPa")
        variables.append(f"geopotential_height_{hpa}hPa")

    return variables


def _build_output_variables() -> list[str]:
    """Atlas variable names produced by this diagnostic."""
    # Must match the Atlas VARIABLES list exactly.
    variables: list[str] = []

    variables.extend(["u10m", "v10m", "u100m", "v100m"])
    variables.extend(["t2m", "sp", "msl", "tcwv"])

    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        variables.append(f"u{hpa}")
    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        variables.append(f"v{hpa}")
    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        variables.append(f"z{hpa}")
    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        variables.append(f"t{hpa}")
    for hpa in ATLAS_PRESSURE_LEVELS_HPA:
        variables.append(f"q{hpa}")

    variables.extend(["sst", "tp"])

    return variables


INPUT_VARIABLES: list[str] = _build_input_variables()
OUTPUT_VARIABLES: list[str] = _build_output_variables()


class MetOfficeToAtlasDiagnostic(torch.nn.Module):
    """Diagnostic model converting native Met Office fields to Atlas inputs.

    Derivations performed:

    - **Wind components**: ``u = -speed × sin(dir)``,
      ``v = -speed × cos(dir)`` for 10 m and all pressure levels.
      Meteorological convention: direction is "from", clockwise from north.
    - **Specific humidity**: from relative humidity + temperature + pressure
      via Bolton (1980) saturation vapour pressure.
    - **Geopotential**: ``z = height × 9.80665``.
    - **100 m wind**: falls back to 10 m (not available from Met Office).
    - **Surface pressure**: approximated by MSLP.
    - **TCWV**: filled with zeros (not available from Met Office).

    The module is stateless and operates in inference mode.
    """

    def __init__(self) -> None:
        super().__init__()

        self.in_variables = np.array(INPUT_VARIABLES)
        self.out_variables = np.array(OUTPUT_VARIABLES)

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
        """Input coordinate system: native Met Office variables."""
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
        """Output coordinate system: Atlas variables."""
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = self.out_variables.copy()
        return output_coords

    def _in(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Select a single variable from the input tensor's variable dim."""
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
        """Saturation vapour pressure (Pa) from temperature (K) via Bolton (1980)."""
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
        """Transform native Met Office fields to Atlas inputs.

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

        out[:, self._out_idx["sst"]] = self._in(x, "surface_temperature")
        out[:, self._out_idx["tp"]] = self._in(x, "lwe_precipitation_rate")

        return out, output_coords
