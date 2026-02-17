"""Example: Run NVIDIA Atlas inference initialised with Met Office T+0 forecast.

This script demonstrates the decomposed Met Office data pipeline:

1. **PlanetaryComputerMetOfficeNative** — pure DataSource that loads raw
   Met Office fields on the native ~0.09° grid with native variable names.
2. **NOAA OISST** — real sea surface temperature from Planetary Computer.
3. **MetOfficeToAtlasDiagnostic** — DiagnosticModel (torch.nn.Module) that
   derives Atlas input variables from native Met Office fields + SST.
4. **fetch_data** with ``interp_to`` — the framework handles regridding from
   native grids to the Atlas 0.25° grid.

Requirements:
    - GPU with sufficient VRAM for Atlas (~16GB+)
    - Internet access for Met Office data, OISST, and Atlas model download

Usage:
    uv run python example_atlas_metoffice.py
"""

from datetime import datetime

import numpy as np
import torch

from earth2studio.data.utils import fetch_data
from earth2studio.models.px.atlas import Atlas

from metoffice_diagnostic import (
    MetOfficeToAtlasDiagnostic,
    combine_inputs,
    fetch_oisst,
)
from metoffice_native import PlanetaryComputerMetOfficeNative


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
    print(f"\nFetching T+0 state for {init_time}...")
    time_array = np.array([np.datetime64(init_time)])
    x_t0, coords_t0 = fetch_data(
        source=native_ds_t0,
        time=time_array,
        variable=metoffice_variables,
        device=device,
        interp_to=atlas_input_coords,
    )
    print(f"  Met Office shape: {x_t0.shape}")

    # ---- T-6h state from previous model run ----
    init_time_minus_6 = datetime(2026, 2, 16, 18)
    time_array_m6 = np.array([np.datetime64(init_time_minus_6)])

    print(f"Fetching T-6h state from {init_time_minus_6} +6h forecast...")
    x_tm6, coords_tm6 = fetch_data(
        source=native_ds_t6,
        time=time_array_m6,
        variable=metoffice_variables,
        device=device,
        interp_to=atlas_input_coords,
    )
    print(f"  Met Office shape: {x_tm6.shape}")

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
    print(f"  Atlas variables shape: {x_t0_atlas.shape}")
    print(f"  Variables: {list(coords_t0_atlas['variable'][:8])}...")

    # Atlas expects lead_time dim: [T-6h, T+0]
    print("\n=== Building Atlas input ===")
    x_input = torch.cat([x_tm6_atlas, x_t0_atlas], dim=1)
    print(f"Input tensor shape: {x_input.shape}")

    input_coords = model.input_coords()
    input_coords["batch"] = np.array([0])
    input_coords["time"] = time_array

    print(f"\n=== Running {forecast_steps}-step Atlas forecast ===")
    results = []
    for step, (pred, pred_coords) in enumerate(model.create_iterator(x_input, input_coords)):
        lead_h = int(pred_coords["lead_time"][0] / np.timedelta64(1, "h"))
        print(f"  Step {step}: T+{lead_h}h")
        results.append((pred.cpu(), pred_coords.copy()))
        if step >= forecast_steps:
            break

    print("\n=== Forecast complete ===")
    for pred, coords in results:
        lead_h = int(coords["lead_time"][0] / np.timedelta64(1, "h"))
        var_list = list(coords["variable"])
        t2m_idx = var_list.index("t2m")
        t2m = pred[0, 0, 0, t2m_idx].numpy()
        print(
            f"  T+{lead_h:3d}h: t2m global mean={t2m.mean():.1f}K "
            f"(min={t2m.min():.1f}, max={t2m.max():.1f})"
        )


if __name__ == "__main__":
    main()
