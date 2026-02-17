"""Example: Atlas forecast using ``run.deterministic`` with Met Office data.

This is a simplified alternative to ``example_atlas_metoffice.py`` that uses
:func:`earth2studio.run.deterministic` instead of manually orchestrating the
forecast loop.

The :class:`MetOfficeAtlasDataSource` wrapper handles the full pipeline
(native Met Office fetch → SST → diagnostic derivation → Atlas-ready fields)
so that ``run.deterministic`` can treat it as a single DataSource.

Requirements:
    - GPU with sufficient VRAM for Atlas (~16 GB+)
    - Internet access for Met Office data, SST, and Atlas model download

Usage:
    uv run python example_atlas_metoffice_run.py
"""

import numpy as np
import torch

from earth2studio.io import ZarrBackend
from earth2studio.models.px.atlas import Atlas
from earth2studio.run import deterministic

from metoffice_atlas_datasource import MetOfficeAtlasDataSource


def main() -> None:
    init_time = "2026-02-17T00:00"
    nsteps = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading Atlas model...")
    package = Atlas.load_default_package()
    model = Atlas.load_model(package)
    print("Atlas model loaded.")

    print("\nRunning deterministic forecast...")
    data = MetOfficeAtlasDataSource()
    io = deterministic(
        time=[init_time],
        nsteps=nsteps,
        prognostic=model,
        data=data,
        io=ZarrBackend(),
        device=device,
    )

    # Print forecast summary
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
