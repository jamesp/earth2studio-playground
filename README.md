

## Installation

Followed: https://nvidia.github.io/earth2studio/userguide/about/install.html

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir earth2studio-playground
cd earth2studio-playground/
echo "export UV_PROJECT_ENVIRONMENT=/mnt/tmp/uv_env" >> ~/.bashrc
echo "export UV_CACHE_DIR=/mnt/tmp/uv_cache" >> ~/.bashrc
echo "export EARTH2STUDIO_CACHE=/mnt/tmp/e2s_cache" >> ~/.bashrc
source ~/.bashrc
uv sync
uv run ipython kernel install --user --env VIRTUAL_ENV $UV_PROJECT_ENVIRONMENT --name=atlas
```

