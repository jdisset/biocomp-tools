"""Parameter schema extraction and filtering for biocomp-tuner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
import numpy as np

from biocomptools.logging_config import get_logger

if TYPE_CHECKING:
    from .session import TunerSession

logger = get_logger(__name__)


@dataclass
class RatioInfo:
    cotx_name: str
    tu_names: list[str]


@dataclass
class ParamDescriptor:
    path: str
    display_name: str
    shape: tuple[int, ...]
    category: str

    current_value: list | float = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    layer_name: Optional[str] = None
    param_name: Optional[str] = None
    ratio_info: Optional[list[RatioInfo]] = None

    cotx_group: Optional[str] = None
    tu_name: Optional[str] = None
    ui_type: str = "number"
    step: float = 0.1
    is_ratio_sum_constrained: bool = False

    def to_dict(self) -> dict:
        d = {
            "path": self.path,
            "display_name": self.display_name,
            "shape": list(self.shape),
            "category": self.category,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "layer_name": self.layer_name,
            "param_name": self.param_name,
            "cotx_group": self.cotx_group,
            "tu_name": self.tu_name,
            "ui_type": self.ui_type,
            "step": self.step,
        }
        if self.ratio_info:
            d["ratio_info"] = [
                {"cotx_name": r.cotx_name, "tu_names": r.tu_names} for r in self.ratio_info
            ]
        return d


@dataclass
class ParamGroup:
    """Group of related parameters for compact UI display."""

    group_id: str
    group_name: str
    category: str
    params: list[ParamDescriptor]
    is_ratio_group: bool = False
    cotx_name: Optional[str] = None
    tu_names: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "category": self.category,
            "params": [p.to_dict() for p in self.params],
            "is_ratio_group": self.is_ratio_group,
            "cotx_name": self.cotx_name,
            "tu_names": self.tu_names,
        }


def get_mask_options_count(mask: np.ndarray) -> int:
    """Count options in a mask.

    For 1D masks: returns count of True values.
    For 2D masks: returns max count of True values per row.
    """
    mask = np.asarray(mask)
    if mask.ndim == 1:
        return int(np.sum(mask))
    return int(np.max(np.sum(mask, axis=1)))


def is_ratio_param(path: str) -> bool:
    return "ratio" in path.lower() and "quantization" not in path.lower()


def is_embedding_param(path: str) -> bool:
    return any(ind in path.lower() for ind in ["embedding", "tl_rate"])


def is_bias_param(path: str) -> bool:
    return "bias" in path.lower()


NON_GRAD_PARAMS = [
    "quantization_mask",
    "node_network_ids",
    "output_tu_indices",
    "input_tu_indices",
    "random_variable_id",
    "min_value",
    "max_value",
    "scale",
]


def is_editable(path: str) -> bool:
    if any(x in path for x in NON_GRAD_PARAMS):
        return False
    if is_ratio_param(path) or is_embedding_param(path) or is_bias_param(path):
        return True
    return False


def categorize_param(path: str) -> str:
    if is_ratio_param(path):
        return "ratios"
    if is_embedding_param(path):
        return "embeddings"
    if is_bias_param(path):
        return "bias"
    return "other"


def make_display_name(path: str) -> str:
    parts = path.split("/")
    display_parts = []
    for part in parts:
        if part == "local":
            continue
        if part.startswith("layer_"):
            display_parts.append(f"Layer {part.replace('layer_', '')}")
        else:
            display_parts.append(part.replace("_", " ").title())
    return " › ".join(display_parts) if display_parts else path


def extract_layer_info(path: str) -> tuple[Optional[str], Optional[str]]:
    parts = path.split("/")
    layer_name = next((p for p in parts if p.startswith("layer_")), None)
    param_name = parts[-1] if parts else None
    return layer_name, param_name


def get_bounds_for_category(category: str) -> tuple[Optional[float], Optional[float], float]:
    if category == "ratios":
        return (1.0, 150.0, 0.1)
    if category == "bias":
        return (0.0, 0.8, 0.01)
    if category == "embeddings":
        return (-5.0, 5.0, 0.1)
    return (None, None, 0.1)


def _build_ratio_metadata(session: TunerSession) -> dict[str, list[RatioInfo]]:
    """Build mapping from namespace to ratio metadata (cotx names, TU names)."""
    stack = session.network_model.stack
    metadata: dict[str, list[RatioInfo]] = {}

    for i, layer in enumerate(stack.layers):
        type_name = layer.type_str()
        if "aggregation" not in type_name.lower() or "inv" in type_name.lower():
            continue

        ns = stack.get_layer_namespace(i)
        ratio_infos = []
        for node in layer.nodes:
            full_node = node.get(stack)
            extra = full_node.extra
            cotx_name = extra.get("cotx_group", "unknown")
            members_data = extra.get("members", {})
            tu_names = sorted(members_data.keys()) if isinstance(members_data, dict) else []
            ratio_infos.append(RatioInfo(cotx_name=cotx_name, tu_names=tu_names))
        metadata[ns] = ratio_infos

    return metadata


def extract_editable_params(session: TunerSession) -> list[ParamDescriptor]:
    assert session.network_model is not None

    ratio_metadata = _build_ratio_metadata(session)
    descriptors = []
    paths_and_values = sorted(session.local_params.data.iter_leaves(), key=lambda x: str(x[0]))

    for path, value in paths_and_values:
        path = str(path)
        if not isinstance(value, (np.ndarray, jnp.ndarray)):
            continue
        if not is_editable(path):
            continue

        value_np = np.asarray(value)
        category = categorize_param(path)
        layer_name, param_name = extract_layer_info(path)
        min_val, max_val, step = get_bounds_for_category(category)

        if is_ratio_param(path):
            ns = path.rsplit("/ratios", 1)[0]
            ratio_info = ratio_metadata.get(ns)
            if ratio_info:
                for cotx_idx, ri in enumerate(ratio_info):
                    for tu_idx, tu_name in enumerate(ri.tu_names):
                        descriptors.append(
                            ParamDescriptor(
                                path=f"{path}[{cotx_idx}][{tu_idx}]",
                                display_name=f"{ri.cotx_name} › {tu_name}",
                                shape=(1,),
                                category=category,
                                current_value=float(value_np[cotx_idx, tu_idx]),
                                min_value=min_val,
                                max_value=max_val,
                                layer_name=layer_name,
                                param_name=param_name,
                                cotx_group=ri.cotx_name,
                                tu_name=tu_name,
                                ui_type="slider",
                                step=step,
                                is_ratio_sum_constrained=False,
                            )
                        )
                continue

        if category in ("bias", "embeddings") and value_np.size > 0:
            flat = value_np.flatten()
            for idx in range(flat.size):
                indices = np.unravel_index(idx, value_np.shape)
                idx_str = "".join(f"[{i}]" for i in indices)
                descriptors.append(
                    ParamDescriptor(
                        path=f"{path}{idx_str}",
                        display_name=f"{make_display_name(path)} [{idx}]",
                        shape=(1,),
                        category=category,
                        current_value=float(flat[idx]),
                        min_value=min_val,
                        max_value=max_val,
                        layer_name=layer_name,
                        param_name=param_name,
                        ui_type="slider",
                        step=step,
                    )
                )
        else:
            descriptors.append(
                ParamDescriptor(
                    path=path,
                    display_name=make_display_name(path),
                    shape=value_np.shape,
                    category=category,
                    current_value=value_np.tolist(),
                    min_value=min_val,
                    max_value=max_val,
                    layer_name=layer_name,
                    param_name=param_name,
                )
            )

    logger.info(f"Extracted {len(descriptors)} editable parameters")
    return descriptors


def group_params_by_category(
    descriptors: list[ParamDescriptor],
) -> dict[str, list[ParamDescriptor]]:
    grouped: dict[str, list[ParamDescriptor]] = {}
    for desc in descriptors:
        grouped.setdefault(desc.category, []).append(desc)
    return grouped


def extract_grouped_params(session: TunerSession) -> list[ParamGroup]:
    """Extract parameters grouped by CoTx/layer for compact display."""
    descriptors = extract_editable_params(session)

    ratio_groups: dict[str, list[ParamDescriptor]] = {}
    other_params: list[ParamDescriptor] = []

    for desc in descriptors:
        if desc.category == "ratios" and desc.cotx_group:
            ratio_groups.setdefault(desc.cotx_group, []).append(desc)
        else:
            other_params.append(desc)

    groups: list[ParamGroup] = []

    for cotx_name, params in ratio_groups.items():
        tu_names = [p.tu_name for p in params if p.tu_name]
        groups.append(
            ParamGroup(
                group_id=f"ratios_{cotx_name}",
                group_name=f"CoTx: {cotx_name}",
                category="ratios",
                params=params,
                is_ratio_group=True,
                cotx_name=cotx_name,
                tu_names=tu_names,
            )
        )

    by_category: dict[str, list[ParamDescriptor]] = {}
    for desc in other_params:
        by_category.setdefault(desc.category, []).append(desc)

    for cat, params in by_category.items():
        groups.append(
            ParamGroup(
                group_id=f"{cat}_all",
                group_name=cat.upper(),
                category=cat,
                params=params,
            )
        )

    return groups
