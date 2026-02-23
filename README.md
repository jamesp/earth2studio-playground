## Running Atlas with Met Office Analysis

WIP. Still not working correctly. Issues:
- zeros and aliasing in the output. Assume this is due to issues with rolling longitude and regridding Arakawa C-Grid onto Atlas gridpoints.
- long inference time. currently running at around 200s/it on a single H100.

### How it works
1. `MetOfficePlanetaryComputer` provides a E2S data source over the Met Office 10km atmospheric STAC catalogue in Planetary Computer.
2. `MetOfficeASDI` provides a E2S data source over the Met Office ocean model .nc files in ASDI.
3. `MetOfficeAtlasDiagnostic` uses a `DiagnosticModel` interface to rename, transform and regrid Met Office data to look like ERA5 data used to train Atlas.
4. `MetOfficeAtlasSource` uses the diagnostic model and wraps in the data source interface so that it can be used as an input data source to `earth2studio.models.px.atlas.Atlas`.

## Installation

Followed: https://nvidia.github.io/earth2studio/userguide/about/install.html. For running on a compute node in Azure ML:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/jamesp/earth2studio-playground
cd earth2studio-playground
echo "export UV_PROJECT_ENVIRONMENT=/mnt/tmp/uv_env" >> ~/.bashrc
echo "export UV_CACHE_DIR=/mnt/tmp/uv_cache" >> ~/.bashrc
echo "export EARTH2STUDIO_CACHE=/mnt/tmp/e2s_cache" >> ~/.bashrc
source ~/.bashrc
uv sync
uv run ipython kernel install --user --env VIRTUAL_ENV $UV_PROJECT_ENVIRONMENT --name=atlas
```
