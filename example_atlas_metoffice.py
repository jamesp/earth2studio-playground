"""Example: Run NVIDIA Atlas inference initialised with Met Office T+0 forecast.

This script demonstrates the decomposed Met Office data pipeline:

1. **PlanetaryComputerMetOfficeNative** — pure DataSource that loads raw
   Met Office fields on the native ~0.09° grid with native variable names.
2. **NOAA OISST** — real sea surface temperature from Planetary Computer.
3. **MetOfficeToAtlasDiagnostic** — DiagnosticModel (torch.nn.Module) that
   derives Atlas input variables from native Met Office fields + SST.
4. **fetch_data** with ``interp_to`` — the framework handles regridding from
   native grids to the Atlas 0.25° grid.

Atlas expects input shape ``(batch, time, lead_time=2, variable=75, lat=721, lon=1440)``
with lead_time ``[-6h, 0h]``.

Requirements:
    - GPU with sufficient VRAM for Atlas (~16GB+)
    - Internet access for Met Office data, OISST, and Atlas model download

Usage:
    uv run python example_atlas_metoffice.py
"""

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from earth2studio.data.utils import fetch_data
from earth2studio.models.px.atlas import Atlas

from coords import interp_coords_to_latlon, make_interp_to
from metoffice_diagnostic import (
    MetOfficeToAtlasDiagnostic,
    combine_inputs,
    fetch_oisst,
)
from metoffice_native import PlanetaryComputerMetOfficeNative


def results_to_dataset(
    results: list[tuple[torch.Tensor, OrderedDict]],
) -> xr.Dataset:
    """Convert Atlas forecast results to an xarray Dataset.

    Each result is ``(pred, pred_coords)`` where pred has shape
    ``(batch, time, lead_time=1, variable, lat, lon)`` and coords
    contain the corresponding arrays.

    The output Dataset has one DataArray per variable, with dimensions
    ``(time, lead_time, lat, lon)``.  The ``lead_time`` coordinate
    stores timedeltas relative to the initialization time.

    Parameters
    ----------
    results : list[tuple[torch.Tensor, CoordSystem]]
        Output from Atlas ``create_iterator``, collected into a list.

    Returns
    -------
    xr.Dataset
        Dataset with forecast fields keyed by Atlas variable name.
    """
    # Stack all steps along lead_time
    lead_times = []
    data_per_step = []
    for pred, coords in results:
        # pred: (batch, time, lead_time=1, variable, lat, lon)
        # Squeeze batch and time (single init), keep lead_time=1
        data_per_step.append(pred[0, 0, 0].numpy())  # (variable, lat, lon)
        lead_times.append(coords["lead_time"][0])

    # Use coords from the first result for spatial/variable metadata
    ref_coords = results[0][1]
    variables = list(ref_coords["variable"])
    lat = ref_coords["lat"]
    lon = ref_coords["lon"]
    time_val = ref_coords["time"]

    stacked = np.stack(data_per_step, axis=0)  # (lead_time, variable, lat, lon)
    lead_time_arr = np.array(lead_times)

    data_vars = {}
    for i, var_name in enumerate(variables):
        data_vars[var_name] = xr.DataArray(
            stacked[:, i, :, :],
            dims=["lead_time", "lat", "lon"],
            attrs={"long_name": var_name},
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time_val,
            "lead_time": lead_time_arr,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "source": "NVIDIA Atlas initialised from Met Office Global 10km Deterministic",
            "history": f"Created by example_atlas_metoffice.py",
        },
    )
    return ds


def fetch_metoffice_regridded(
    native_ds: PlanetaryComputerMetOfficeNative,
    time_array: np.ndarray,
    variables: np.ndarray,
    *,
    atlas_input_coords: OrderedDict,
    device: torch.device,
) -> tuple[torch.Tensor, OrderedDict]:
    """Fetch Met Office data and regrid to the Atlas 0.25° grid.

    Handles the ``_lat``/``_lon`` → ``lat``/``lon`` key translation that
    ``fetch_data`` with ``interp_to`` requires.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tensor of shape ``(time, lead_time=1, variable, lat, lon)`` and
        coordinates with ``lat``/``lon`` keys.
    """
    data, coords = fetch_data(
        source=native_ds,
        time=time_array,
        variable=variables,
        device=device,
        interp_to=make_interp_to(atlas_input_coords),
    )
    return data, interp_coords_to_latlon(coords)


def main():
    init_time = datetime(2026, 2, 17, 0)
    forecast_steps = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== Loading Atlas model ===")
    package = Atlas.load_default_package()
    model = Atlas.load_model(package)
    model = model.to(device)
    model.eval()
    print("Atlas model loaded successfully.")

    print("\n=== Setting up Met Office data pipeline ===")
    native_ds_t0 = PlanetaryComputerMetOfficeNative(forecast_hour=0, verbose=True)
    native_ds_t6 = PlanetaryComputerMetOfficeNative(forecast_hour=6, verbose=True)
    diagnostic = MetOfficeToAtlasDiagnostic().to(device)
    metoffice_variables = diagnostic.metoffice_variables
    print(f"Met Office variables: {len(metoffice_variables)}")
    print(f"Atlas output variables: {len(diagnostic.out_variables)}")

    atlas_input_coords = model.input_coords()

    # ---- T+0 state ----
    time_array = np.array([np.datetime64(init_time)])
    print(f"\nFetching T+0 state for {init_time}...")
    x_t0, coords_t0 = fetch_metoffice_regridded(
        native_ds_t0, time_array, metoffice_variables,
        atlas_input_coords=atlas_input_coords, device=device,
    )
    print(f"  Met Office regridded shape: {x_t0.shape}")
    print(f"  Coord keys: {list(coords_t0.keys())}")

    # ---- T-6h state from previous model run ----
    init_time_minus_6 = datetime(2026, 2, 16, 18)
    time_array_m6 = np.array([np.datetime64(init_time_minus_6)])
    print(f"Fetching T-6h state from {init_time_minus_6} +6h forecast...")
    x_tm6, coords_tm6 = fetch_metoffice_regridded(
        native_ds_t6, time_array_m6, metoffice_variables,
        atlas_input_coords=atlas_input_coords, device=device,
    )
    print(f"  Met Office regridded shape: {x_tm6.shape}")

    # ---- Fetch SST from OISST ----
    print("\n=== Fetching NOAA OISST sea surface temperature ===")
    sst_t0, sst_coords_t0 = fetch_oisst(
        time_array, atlas_input_coords=atlas_input_coords, device=device
    )
    sst_tm6, sst_coords_tm6 = fetch_oisst(
        time_array_m6, atlas_input_coords=atlas_input_coords, device=device
    )
    print(f"  SST shape: {sst_t0.shape}")

    # ---- Combine Met Office + SST, then run diagnostic ----
    print("\n=== Combining inputs and running diagnostic ===")
    x_t0_combined, coords_t0_combined = combine_inputs(
        x_t0, coords_t0, sst_t0, sst_coords_t0
    )
    x_tm6_combined, coords_tm6_combined = combine_inputs(
        x_tm6, coords_tm6, sst_tm6, sst_coords_tm6
    )
    print(f"  Combined input shape: {x_t0_combined.shape}")

    x_t0_atlas, coords_t0_atlas = diagnostic(x_t0_combined, coords_t0_combined)
    x_tm6_atlas, coords_tm6_atlas = diagnostic(x_tm6_combined, coords_tm6_combined)
    print(f"  Atlas variables shape (per timestep): {x_t0_atlas.shape}")
    print(f"  Variables: {list(coords_t0_atlas['variable'][:8])}...")

    # ---- Build Atlas input: (batch, time, lead_time=2, variable, lat, lon) ----
    #
    # Atlas expects lead_time dim with [-6h, 0h] (sliding window of two snapshots).
    #
    # After diagnostic, each timestep tensor has shape:
    #   (time=1, lead_time=1, variable=75, lat=721, lon=1440)
    # with coords {time, lead_time, variable, lat, lon}.
    #
    # We concatenate along the lead_time dim (dim 1) to combine the two
    # snapshots, then prepend a batch dim.
    print("\n=== Building Atlas input ===")

    lead_time_dim = list(coords_t0_atlas.keys()).index("lead_time")
    # Concatenate t-6h and t+0 along lead_time: (time=1, lead_time=2, 75, 721, 1440)
    x_input = torch.cat([x_tm6_atlas, x_t0_atlas], dim=lead_time_dim)
    # Add batch dim: (batch=1, time=1, lead_time=2, 75, 721, 1440)
    x_input = x_input.unsqueeze(0)
    print(f"  Input tensor shape: {x_input.shape}")
    print(f"  Expected:           (1, 1, 2, 75, 721, 1440)")

    input_coords = model.input_coords()
    input_coords["batch"] = np.array([0])
    input_coords["time"] = time_array

    # Verify shapes match Atlas expectations:
    # (batch, time, lead_time, variable, lat, lon) = (1, 1, 2, 75, 721, 1440)
    assert x_input.dim() == 6, f"Expected 6D tensor, got {x_input.dim()}D"
    assert x_input.shape[2] == 2, (
        f"lead_time dim should be 2, got {x_input.shape[2]}"
    )
    assert x_input.shape[3] == len(input_coords["variable"]), (
        f"variable mismatch: tensor has {x_input.shape[3]}, "
        f"coords has {len(input_coords['variable'])}"
    )

    print(f"\n=== Running {forecast_steps}-step Atlas forecast ===")
    results = []
    for step, (pred, pred_coords) in enumerate(
        model.create_iterator(x_input, input_coords)
    ):
        lead_h = int(pred_coords["lead_time"][0] / np.timedelta64(1, "h"))
        print(f"  Step {step}: T+{lead_h}h")
        results.append((pred.cpu(), pred_coords.copy()))
        if step >= forecast_steps:
            break

    print("\n=== Forecast summary ===")
    for pred, coords in results:
        lead_h = int(coords["lead_time"][0] / np.timedelta64(1, "h"))
        var_list = list(coords["variable"])
        t2m_idx = var_list.index("t2m")
        t2m = pred[0, 0, 0, t2m_idx].numpy()
        print(
            f"  T+{lead_h:3d}h: t2m global mean={t2m.mean():.1f}K "
            f"(min={t2m.min():.1f}, max={t2m.max():.1f})"
        )

    # ---- Save forecast to zarr ----
    output_path = Path(
        f"forecast_metoffice_atlas_{init_time:%Y%m%d_%H%M}.zarr"
    )
    print(f"\n=== Saving forecast to {output_path} ===")
    ds = results_to_dataset(results)
    ds.to_zarr(output_path, mode="w")
    print(f"  Wrote {output_path} ({sum(v.nbytes for v in ds.data_vars.values()) / 1e6:.1f} MB)")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.sizes)}")


if __name__ == "__main__":
    main()
