from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from biocomp.jaxutils import tree_get

from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class CommitCache:
    """Cache of committed networks, scoped to a specific stack instance.

    Thread-safe. Keyed by (rep_id, target_id). Only returns cached
    results when the caller's stack matches the one used to populate
    the cache (identity check via `id(stack)`).
    """

    def __init__(self, stack: Any) -> None:
        self._stack_id = id(stack)
        self._cache: dict[tuple[int, int], list[Any]] = {}
        self._lock = threading.Lock()

    def get(self, rep_id: int, target_id: int, stack: Any) -> list[Any] | None:
        if id(stack) != self._stack_id:
            return None
        with self._lock:
            return self._cache.get((rep_id, target_id))

    def put(self, rep_id: int, target_id: int, networks: list[Any], stack: Any) -> None:
        assert id(stack) == self._stack_id, "cannot cache results from a different stack"
        with self._lock:
            self._cache[(rep_id, target_id)] = networks

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


@dataclass(frozen=True)
class CommitRequest:
    rep_id: int
    target_id: int
    net_id: int
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitResult:
    request: CommitRequest
    network: Any | None
    error: str | None = None


def precommit_pairs(
    full_params: Any,
    stack: Any,
    pairs: set[tuple[int, int]],
    *,
    fail_fast: bool = False,
    fill_failures_with: Callable[[int, int], list[Any]] | None = None,
    on_before: Callable[[int, int, Any, Any], None] | None = None,
    on_after: Callable[[int, int, Any, Any, list[Any]], None] | None = None,
    on_error: Callable[[int, int, Any, Any, Exception], None] | None = None,
    commit_cache: CommitCache | None = None,
    parallel: bool = False,
    max_workers: int = 4,
) -> tuple[dict[tuple[int, int], list[Any]], list[tuple[int, int, str]]]:
    local_cache: dict[tuple[int, int], list[Any]] = {}
    failures: list[tuple[int, int, str]] = []

    # Filter out pairs already in the shared cache
    remaining = sorted(pairs)
    if commit_cache is not None:
        filtered = []
        for rep_id, target_id in remaining:
            cached = commit_cache.get(rep_id, target_id, stack)
            if cached is not None:
                local_cache[(rep_id, target_id)] = cached
            else:
                filtered.append((rep_id, target_id))
        if filtered != remaining:
            logger.info(
                f"CommitCache hit: {len(remaining) - len(filtered)}/{len(remaining)} pairs cached, "
                f"{len(filtered)} remaining"
            )
        remaining = filtered

    if not remaining:
        return local_cache, failures

    def _commit_one(rep_id: int, target_id: int) -> tuple[int, int, list[Any] | None, str | None]:
        bparams = tree_get(full_params, (rep_id, target_id))
        assert bparams is not None, f"missing params for (rep={rep_id}, target={target_id})"
        if on_before is not None:
            on_before(rep_id, target_id, bparams, stack)
        try:
            committed_networks = stack.commit(bparams)
            if on_after is not None:
                on_after(rep_id, target_id, bparams, stack, committed_networks)
            return rep_id, target_id, committed_networks, None
        except Exception as exc:
            if on_error is not None:
                on_error(rep_id, target_id, bparams, stack, exc)
            return rep_id, target_id, None, f"{type(exc).__name__}: {exc}"

    t0 = time.perf_counter()

    if parallel and len(remaining) > 1:
        workers = min(max_workers, len(remaining))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_commit_one, r, t): (r, t) for r, t in remaining
            }
            for future in as_completed(futures):
                rep_id, target_id, networks, error = future.result()
                if error is not None:
                    failures.append((rep_id, target_id, error))
                    if fill_failures_with is not None:
                        local_cache[(rep_id, target_id)] = fill_failures_with(rep_id, target_id)
                    if fail_fast:
                        raise RuntimeError(error)
                else:
                    assert networks is not None
                    local_cache[(rep_id, target_id)] = networks
    else:
        for rep_id, target_id in remaining:
            _, _, networks, error = _commit_one(rep_id, target_id)
            if error is not None:
                failures.append((rep_id, target_id, error))
                if fill_failures_with is not None:
                    local_cache[(rep_id, target_id)] = fill_failures_with(rep_id, target_id)
                if fail_fast:
                    raise RuntimeError(error)
            else:
                assert networks is not None
                local_cache[(rep_id, target_id)] = networks

    elapsed = time.perf_counter() - t0
    mode = f"parallel({min(max_workers, len(remaining))}w)" if parallel and len(remaining) > 1 else "serial"
    logger.debug(f"Committed {len(remaining)} pairs in {elapsed:.2f}s ({mode})")

    # Populate shared cache with freshly committed results
    if commit_cache is not None:
        for key, networks in local_cache.items():
            if commit_cache.get(key[0], key[1], stack) is None:
                commit_cache.put(key[0], key[1], networks, stack)

    return local_cache, failures


def resolve_commit_requests(
    requests: list[CommitRequest],
    commit_cache: dict[tuple[int, int], list[Any]],
) -> list[CommitResult]:
    results: list[CommitResult] = []
    for req in requests:
        committed = commit_cache.get((req.rep_id, req.target_id))
        if committed is None:
            results.append(
                CommitResult(
                    request=req,
                    network=None,
                    error=f"missing commit result for (rep={req.rep_id}, target={req.target_id})",
                )
            )
            continue
        if not (0 <= req.net_id < len(committed)):
            results.append(
                CommitResult(
                    request=req,
                    network=None,
                    error=(
                        f"net_id {req.net_id} out of bounds for committed list size {len(committed)} "
                        f"(rep={req.rep_id}, target={req.target_id})"
                    ),
                )
            )
            continue
        results.append(CommitResult(request=req, network=committed[req.net_id], error=None))
    return results


def make_design_input(
    *,
    network: Any,
    target: Any,
    target_name: str,
    rank: int,
    replicate: int,
    net_id: int,
    loss: float,
    run_name: str,
    design_dir: str,
    recipe_hash: str | None = None,
) -> Any:
    from biocomptools.toollib.design_eval import DesignInput
    from biocomptools.toollib.hashutils import pronounceable_hash48

    fingerprint = recipe_hash
    if fingerprint is None:
        fingerprint = pronounceable_hash48(f"{target_name}_{rank}_{replicate}_{net_id}".encode())
    return DesignInput(
        network=network,
        target=target,
        target_name=target_name,
        rank=rank,
        replicate=replicate,
        scaffold_network_name=getattr(network, 'name', f"net_{net_id}"),
        loss=float(loss),
        recipe_hash=fingerprint,
        run_name=run_name,
        design_dir=design_dir,
    )


def evaluate_design_inputs(model: Any, design_inputs: list[Any], *, max_evals: int) -> list[Any]:
    from biocomptools.toollib.design_eval import DesignEvaluator

    if not design_inputs:
        return []
    evaluator = DesignEvaluator(model, max_evals=max_evals, fail_fast=True)
    return evaluator.evaluate_designs(design_inputs)


def serialize_network_recipe(
    network: Any,
    *,
    auto_name_from_l1: bool = False,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, str, str]:
    import dracon
    import yaml
    from biocomptools.toollib.hashutils import pronounceable_hash48

    recipe = network.to_recipe(auto_name_from_l1=auto_name_from_l1)
    recipe_yaml = dracon.dump(recipe)
    recipe_hash = pronounceable_hash48(recipe_yaml.encode('utf-8'))
    if metadata:
        metadata_with_hash = dict(metadata)
        metadata_with_hash.setdefault('recipe_hash', recipe_hash)
        metadata_yaml = yaml.dump({'_metadata': metadata_with_hash}, default_flow_style=False)
        full_yaml = metadata_yaml + '\n' + recipe_yaml
    else:
        full_yaml = recipe_yaml
    return recipe, full_yaml, recipe_hash


def save_network_recipe_yaml(
    network: Any,
    output_path: str | Path,
    *,
    auto_name_from_l1: bool = False,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    recipe, full_yaml, recipe_hash = serialize_network_recipe(
        network,
        auto_name_from_l1=auto_name_from_l1,
        metadata=metadata,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_yaml)
    return recipe, recipe_hash


def build_design_result(
    evaluated_design: Any,
    *,
    model: Any,
    fingerprint: str | None = None,
) -> Any:
    from biocomptools.toollib.figuremakers.designutils import DesignResult

    inp = evaluated_design.input
    return DesignResult(
        network=inp.network,
        target=inp.target,
        target_name=inp.target_name,
        rank=inp.rank,
        replicate=inp.replicate,
        scaffold_network_name=inp.scaffold_network_name,
        loss=inp.loss,
        recipe_hash=inp.recipe_hash,
        run_name=inp.run_name,
        model=model,
        gt_data=evaluated_design.gt_data,
        pred_data=evaluated_design.pred_data,
        lattice_data=evaluated_design.lattice_data,
        lattice_grid=evaluated_design.lattice_grid,
        lattice_extent=evaluated_design.lattice_extent,
        lattice_resolution=evaluated_design.lattice_resolution,
        exp_x_data=evaluated_design.exp_x_data,
        fingerprint=fingerprint,
    )


def invoke_design_summary_plot(result: Any, *, output_dir: str | Path) -> None:
    from biocomptools.plot import PlotJob

    invoke_fn = getattr(PlotJob, "invoke", None)
    assert callable(invoke_fn), "PlotJob.invoke is required for design summary plotting"
    invoke_fn(
        'biocomp-jobs/plot/auto_figures/autofig_design_summary.yaml',
        result=result,
        output_dir=str(output_dir),
    )
