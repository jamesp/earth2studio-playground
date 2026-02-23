"""Atlas forecast using ``run.deterministic`` with Met Office data.

Uses :class:`~src.metoffice_source.MetOfficeAtlasSource` to fetch
Atlas-ready initial conditions from the Met Office global deterministic
forecast and ocean SST analysis, then runs Atlas for a short forecast.

Requirements:
    - GPU with sufficient VRAM for Atlas (~16 GB+)
    - Internet access for Met Office data and Atlas model download

Usage::

    uv run python src/run_atlas_forecast.py
"""

import sys
import os

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from earth2studio.io import ZarrBackend
from earth2studio.models.px.atlas import Atlas
from earth2studio.run import deterministic

from src.metoffice_source import MetOfficeAtlasSource


def main() -> None:
    init_time = "2026-01-24T00:00"
    nsteps = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading Atlas model...")
    package = Atlas.load_default_package()
    model = Atlas.load_model(package)
    print("Atlas model loaded.")

    print("\nRunning deterministic forecast...")
    data = MetOfficeAtlasSource()
    io = deterministic(
        time=[init_time],
        nsteps=nsteps,
        prognostic=model,
        data=data,
        io=ZarrBackend(f"/mnt/tmp/data/forecast_{init_time.replace(':', '-')}.zarr"),
        device=device,
    )

    print("\n=== Forecast summary ===")
    lead_times = io["lead_time"]
    lat = io["lat"]
    lon = io["lon"]
    print(f"Lead times: {lead_times.shape[0]} steps")
    print(f"Grid: {lat.shape[0]} x {lon.shape[0]}")

    t2m = io["t2m"]  # shape: (time, lead_time, lat, lon)
    for i, lt in enumerate(lead_times):
        lead_h = int(lt / np.timedelta64(1, "h"))
        vals = t2m[0, i]
        print(
            f"  T+{lead_h:3d}h: t2m mean={vals.mean():.1f}K "
            f"(min={vals.min():.1f}, max={vals.max():.1f})"
        )


if __name__ == "__main__":
    main()
