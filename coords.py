# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""Coordinate helpers for bridging earth2studio's spatial key conventions.

earth2studio uses two different key names for latitude/longitude in
:class:`~earth2studio.utils.type.CoordSystem` dicts:

- **"lat" / "lon"** — used by models (Atlas, Pangu, etc.) in their
  ``input_coords()`` / ``output_coords()``.
- **"_lat" / "_lon"** — required by :func:`~earth2studio.data.utils.fetch_data`
  in the ``interp_to`` argument, and produced in the output coords when
  ``interp_to`` is used.

This module provides thin helpers to translate between the two conventions
so that data-fetching and model-feeding code stays readable.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from earth2studio.utils.type import CoordSystem


def make_interp_to(model_coords: CoordSystem) -> CoordSystem:
    """Build an ``interp_to`` dict from a model's coordinate system.

    :func:`~earth2studio.data.utils.fetch_data` expects the interpolation
    target to use ``"_lat"`` / ``"_lon"`` keys, but model ``input_coords()``
    use ``"lat"`` / ``"lon"``.  This helper copies the spatial arrays
    under the right keys.

    Parameters
    ----------
    model_coords : CoordSystem
        A coordinate system containing ``"lat"`` and ``"lon"`` arrays
        (e.g. from ``model.input_coords()``).

    Returns
    -------
    CoordSystem
        ``OrderedDict({"_lat": ..., "_lon": ...})`` suitable for passing
        as ``interp_to`` to ``fetch_data``.
    """
    return OrderedDict(
        {
            "_lat": model_coords["lat"],
            "_lon": model_coords["lon"],
        }
    )


def interp_coords_to_latlon(coords: CoordSystem) -> CoordSystem:
    """Rename ``_lat`` / ``_lon`` keys back to ``lat`` / ``lon``.

    After :func:`~earth2studio.data.utils.fetch_data` with ``interp_to``,
    the output coordinate system uses ``"_lat"`` / ``"_lon"`` for the
    spatial dimensions.  Models expect ``"lat"`` / ``"lon"``.  This
    helper does the rename, preserving key order.

    Parameters
    ----------
    coords : CoordSystem
        Coordinate system potentially containing ``"_lat"`` / ``"_lon"``.

    Returns
    -------
    CoordSystem
        New coordinate system with ``"lat"`` / ``"lon"`` keys.
        If the input already uses ``"lat"`` / ``"lon"``, it is returned
        unchanged (as a copy).
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
