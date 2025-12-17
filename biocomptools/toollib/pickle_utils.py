"""Utilities for pickling JAX arrays and lazy data structures.

These utilities handle serialization challenges with:
- JAX arrays (must be converted to numpy)
- JAX Device objects (cannot be pickled, must be removed)
- LazyPlotData closures (capture unpicklable objects)
"""

import numpy as np


def _detached_get_xy(pdata):
    """Dummy get_xy for detached LazyPlotData - data should already be evaluated."""
    raise RuntimeError(
        "LazyPlotData has been detached for pickling - data should already be evaluated"
    )


def detach_lazy_plot_data(obj, visited=None):
    """Recursively find LazyPlotData objects and detach their closures for pickling.

    LazyPlotData.get_xy closures capture NetworkPrediction instances which hold
    BiocompModel with JAX Device objects that cannot be pickled. This function:
    1. Forces evaluation of lazy data (populates xval/yval)
    2. Replaces get_xy with a dummy function (no closure references)

    After this, the LazyPlotData contains all the data it needs and is picklable.
    """
    from biocomp.plotutils import LazyPlotData

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(obj, LazyPlotData):
        if obj.xval is None:
            try:
                _ = obj.x
                _ = obj.y
            except (ValueError, AttributeError, RuntimeError):
                pass
        obj.get_xy = _detached_get_xy
        return

    try:
        if isinstance(obj, dict):
            for v in obj.values():
                detach_lazy_plot_data(v, visited)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                detach_lazy_plot_data(v, visited)
        elif hasattr(obj, '__dict__'):
            for v in vars(obj).values():
                detach_lazy_plot_data(v, visited)
    except (TypeError, AttributeError):
        pass


def convert_jax_to_numpy_inplace(obj, visited=None):
    """Recursively convert JAX arrays to numpy arrays in-place for pickling.

    Also removes JAX Device objects which cannot be pickled.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    try:
        from jax import Array as JaxArray
        from jax._src.xla_bridge import Device
    except ImportError:
        try:
            from jax import Array as JaxArray

            Device = None
        except ImportError:
            return obj

    if Device is not None and isinstance(obj, Device):
        return None

    if isinstance(obj, JaxArray):
        return np.asarray(obj)

    type_name = type(obj).__module__ + '.' + type(obj).__name__
    if 'jax' in type_name.lower() and 'device' in type_name.lower():
        return None
    if 'jax' in type_name.lower() and 'array' in type_name.lower():
        try:
            return np.asarray(obj)
        except (TypeError, ValueError):
            pass

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            converted = convert_jax_to_numpy_inplace(v, visited)
            if converted is None and v is not None:
                del obj[k]
            else:
                obj[k] = converted
    elif isinstance(obj, list):
        i = 0
        while i < len(obj):
            converted = convert_jax_to_numpy_inplace(obj[i], visited)
            if converted is None and obj[i] is not None:
                obj.pop(i)
            else:
                obj[i] = converted
                i += 1
    elif isinstance(obj, tuple):
        converted = [convert_jax_to_numpy_inplace(v, visited) for v in obj]
        return tuple(
            c
            for c, orig in zip(converted, obj, strict=True)
            if not (c is None and orig is not None)
        )
    elif hasattr(obj, '__dict__'):
        for attr_name in list(vars(obj).keys()):
            try:
                val = getattr(obj, attr_name)
                converted = convert_jax_to_numpy_inplace(val, visited)
                if converted is None and val is not None:
                    setattr(obj, attr_name, None)
                elif converted is not val:
                    setattr(obj, attr_name, converted)
            except (AttributeError, TypeError):
                pass
    elif hasattr(obj, '__slots__'):
        for slot in obj.__slots__:
            try:
                val = getattr(obj, slot)
                converted = convert_jax_to_numpy_inplace(val, visited)
                if converted is None and val is not None:
                    setattr(obj, slot, None)
                elif converted is not val:
                    setattr(obj, slot, converted)
            except (AttributeError, TypeError):
                pass

    return obj
