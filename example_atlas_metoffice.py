"""Example: Run NVIDIA Atlas inference initialised with Met Office T+0 forecast.

This script demonstrates how to use the PlanetaryComputerMetOffice data loader
to fetch initial conditions from the Met Office global deterministic model and
run the Atlas AI weather model for a 24-hour forecast.

Requirements:
    - GPU with sufficient VRAM for Atlas (~16GB+)
    - Internet access for Met Office data and Atlas model download

Usage:
    uv run python example_atlas_metoffice.py
"""

import numpy as np
import torch
from datetime import datetime

from earth2studio.models.px.atlas import Atlas, VARIABLES as ATLAS_VARIABLES
from earth2studio.data.utils import fetch_data
from metoffice_data import PlanetaryComputerMetOffice


def main():
    # Configuration
    init_time = datetime(2026, 2, 17, 0)  # Met Office model run time
    forecast_steps = 4  # 4 steps × 6h = 24 hours
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load the Atlas model ---
    print("\n=== Loading Atlas model ===")
    package = Atlas.load_default_package()
    model = Atlas.load_model(package)
    model = model.to(device)
    model.eval()
    print("Atlas model loaded successfully.")

    # --- Step 2: Fetch Met Office initial conditions ---
    print("\n=== Fetching Met Office initial conditions ===")
    ds = PlanetaryComputerMetOffice(forecast_hour=0, verbose=True)

    # Atlas needs two input lead times: T-6h and T+0
    # For T-6h we use the 6-hour forecast from the same run
    ds_t_minus_6 = PlanetaryComputerMetOffice(forecast_hour=6, verbose=True)

    # Fetch T+0 data
    time_array = np.array([np.datetime64(init_time)])
    variables = np.array(ATLAS_VARIABLES)

    print(f"Fetching T+0 state for {init_time}...")
    x_t0, coords_t0 = fetch_data(ds, time_array, variables, device=device)
    print(f"  Shape: {x_t0.shape}")

    # For T-6h, use the previous model run's T+0 or current run's T+6 from 6h earlier
    # Here we approximate by using the 6h forecast from the 6h-earlier run
    init_time_minus_6 = datetime(2026, 2, 16, 18)  # previous run
    time_array_m6 = np.array([np.datetime64(init_time_minus_6)])

    print(f"Fetching T-6h state from {init_time_minus_6} +6h forecast...")
    x_tm6, coords_tm6 = fetch_data(ds_t_minus_6, time_array_m6, variables, device=device)
    print(f"  Shape: {x_tm6.shape}")

    # --- Step 3: Build Atlas input ---
    # Atlas expects shape: (batch, time, lead_time=2, variable, lat, lon)
    # lead_time[0] = T-6h, lead_time[1] = T+0
    print("\n=== Building Atlas input ===")
    x_input = torch.cat([x_tm6, x_t0], dim=1)  # stack along lead_time dim
    print(f"Input tensor shape: {x_input.shape}")

    # Build input coords
    input_coords = model.input_coords()
    input_coords["batch"] = np.array([0])
    input_coords["time"] = time_array

    # --- Step 4: Run autoregressive forecast ---
    print(f"\n=== Running {forecast_steps}-step Atlas forecast ===")
    results = []
    for step, (pred, pred_coords) in enumerate(model.create_iterator(x_input, input_coords)):
        lead_h = int(pred_coords["lead_time"][0] / np.timedelta64(1, "h"))
        print(f"  Step {step}: T+{lead_h}h")
        results.append((pred.cpu(), pred_coords.copy()))
        if step >= forecast_steps:
            break

    # --- Step 5: Print summary ---
    print("\n=== Forecast complete ===")
    for pred, coords in results:
        lead_h = int(coords["lead_time"][0] / np.timedelta64(1, "h"))
        # Find t2m index
        var_list = list(coords["variable"])
        t2m_idx = var_list.index("t2m")
        t2m = pred[0, 0, 0, t2m_idx].numpy()
        print(
            f"  T+{lead_h:3d}h: t2m global mean={t2m.mean():.1f}K "
            f"(min={t2m.min():.1f}, max={t2m.max():.1f})"
        )


if __name__ == "__main__":
    main()
