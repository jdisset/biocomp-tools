from __future__ import annotations

from biocomptools.toollib.plot import Figure, PlotConfig, load_default_plotconf
from biocomptools.modelmodel import BiocompModel, NetworkModel, load_model
from biocomptools.logging_config import get_logger
from biocomp.plotutils import FigureSpec
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit as Unit
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure as MplFigure
from typing import Annotated, Any, Callable
from pathlib import Path
from pydantic import Field, BeforeValidator
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

logger = get_logger(__name__)


CMAP = "bc_blues"
# CMAP = 'bc_reds'
# CMAP = 'bc_greens'

N_SAMPLES, SHOW_INVERSE, CMAP_TRUNCATE_MIN = 100_000, True, 0.25
INV_OUT_RANGE = (0, 0.7)
DEFAULT_RANGE = (0.01, 0.8)

NODE_PARENTS: dict[str, list[str]] = {
    "inv_output": ["input"],
    "inv_translation": ["inv_output"],
    "inv_transcription": ["inv_translation"],
    "inv_source": ["inv_transcription"],
    "source": ["aggregation", "inv_source"],
    "transcription": ["source"],
    "translation": ["transcription", "sequestron_ern"],
    "sequestron_ern": ["translation", "transcription"],
    "output": ["translation"],
}

TOPO_ORDER = [
    ("inv_output", "inverse"),
    ("inv_translation", "inv_uorf"),
    ("inv_transcription", "inverse"),
    ("inv_source", "inverse"),
    ("inv_source", "inv_src"),
    ("source", "source"),
    ("transcription", "basic"),
    ("translation", "basic"),
    ("translation", "uorf"),
    ("sequestron_ern", "ern"),
    ("output", "output"),
]

FWD_LABELS = {"source": "plasmid → DNA", "transcription": "DNA → mRNA", "translation": "mRNA → PRT"}
INV_LABELS = {
    "inv_output": "Fluo → PRT",
    "inv_source": "DNA → plasmid",
    "inv_transcription": "mRNA → DNA",
    "inv_translation": "PRT → mRNA",
}
BASIC_RECIPE = [[["hEF1a", "eYFP"], ["hEF1a", "eBFP2"]]]


@dataclass(frozen=True)
class ApplyFn:
    single: Callable[..., float]
    batch: Callable[..., np.ndarray]

    def __call__(self, *a, **kw) -> float:
        return self.single(*a, **kw)


@dataclass(frozen=True)
class NodeInfo:
    name: str
    node_type: str
    apply_fn: ApplyFn
    node_id: int
    emb_name: str | None = None
    emb_val: float | tuple[float, ...] | None = None
    emb_index: int | None = None

    @property
    def emb_dim(self) -> int:
        if self.emb_val is None:
            return 0
        if isinstance(self.emb_val, tuple):
            return len(self.emb_val)
        return 1

    @property
    def emb_scalar(self) -> float | None:
        if self.emb_val is None:
            return None
        if isinstance(self.emb_val, tuple):
            return None
        return self.emb_val


def _norm(s: str) -> str:
    return s.lower().replace(" ", "_")


class InnerNodesFigure(Figure):
    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    n_samples: int = N_SAMPLES
    figure_spec: FigureSpec = Field(default_factory=FigureSpec)
    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)
    print_summary: bool = True
    show_distribution: bool = False
    n_trendline_points: int = 200
    max_trajectory_points: int = 1000
    history_dir: Path | None = None
    embedding_trajectories: dict[str, list[tuple[float, ...]]] | None = None

    _cache: dict = {}
    _ranges: dict[str, tuple[float, float]] = {}
    _trajectories: dict[str, list[tuple[float, ...]]] | None = None

    def _cmap(self, truncate: bool = True):
        base = plt.get_cmap(CMAP)
        return LinearSegmentedColormap.from_list(
            "c", base(np.linspace(CMAP_TRUNCATE_MIN if truncate else 0, 1, 256))
        )

    def _emb(self, path: str, names: list[str]) -> dict[str, float | tuple[float, ...]]:
        try:
            v: Any = self.model.shared_params
            for k in path.split("/"):
                v = v[k]
            result: dict[str, float | tuple[float, ...]] = {}
            for n, x in zip(names, v, strict=True):
                arr = np.asarray(x).ravel()
                result[n] = float(arr[0]) if arr.shape[0] == 1 else tuple(float(a) for a in arr)
            return result
        except (KeyError, IndexError, TypeError):
            return {}

    def _get_emb_dim(self, emb_type: str) -> int:
        cc = self.model.compute_config
        if not cc or not cc.node_functions:
            return 1
        dim_key = {
            "affinity": ("sequestron_ERN", "affinity_dim"),
            "tl_rate": ("translation", "rate_dim"),
            "tc_rate": ("transcription", "rate_dim"),
        }
        if emb_type not in dim_key:
            return 1
        node_name, kwarg_name = dim_key[emb_type]
        if node_name not in cc.node_functions:
            return 1
        return int(cc.node_functions[node_name].kwargs.get(kwarg_name, 1))

    def _stack(self, nets: list) -> tuple[Any, Any, Any]:
        key = tuple(id(n) for n in nets)
        if key not in self._cache:
            m = NetworkModel(model=self.model, network=nets)
            self._cache[key] = (m.stack, m.params)
        return *self._cache[key], jax.random.PRNGKey(0)

    def _layer(self, stack, substr: str):
        return next((ly for ly in stack.layers if substr in ly.type_str() and ly.f_apply), None)

    def _apply(self, f, params, key, n_dup: int = 0) -> ApplyFn:
        def call(*inp, nid: int, rv: float):
            if n_dup:
                inp = tuple(inp[0] for _ in range(n_dup))
            r, _ = f(
                *[jnp.atleast_1d(jnp.asarray(x)) for x in inp],
                random_vars=jnp.atleast_1d(jnp.asarray(rv)),
                params=params,
                node_id=jnp.array(nid),
                key=key,
            )
            return r[0] if n_dup else r

        def single(*i, node_id, random_var=0.5):
            return float(jnp.squeeze(call(*i, nid=node_id, rv=random_var)))

        def batch(arr: np.ndarray, node_id: int, random_var: float = 0.5) -> np.ndarray:
            arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
            return np.asarray(
                jax.vmap(lambda *a: jnp.squeeze(call(*a, nid=node_id, rv=random_var)))(
                    *[arr[:, i] for i in range(arr.shape[1])]
                )
            )

        return ApplyFn(single=single, batch=batch)

    def _in_range(self, nt: str) -> tuple[float, float]:
        n = _norm(nt)
        if n == "inv_output":
            return INV_OUT_RANGE
        lo, hi = float("inf"), float("-inf")
        for p in NODE_PARENTS.get(n, []):
            if p in self._ranges:
                lo, hi = min(lo, self._ranges[p][0]), max(hi, self._ranges[p][1])
        if lo == float("inf"):
            return DEFAULT_RANGE
        m = (hi - lo) * 0.02
        return (lo - m, hi + m)

    def _reg_range(self, nt: str, r: tuple[float, float]):
        n = _norm(nt)
        self._ranges[n] = (
            (min(self._ranges[n][0], r[0]), max(self._ranges[n][1], r[1]))
            if n in self._ranges
            else r
        )

    def _eval(self, node: NodeInfo, inp: np.ndarray, rv: float = 0.5) -> np.ndarray:
        r = node.apply_fn.batch(inp, node_id=node.node_id, random_var=rv)
        return (
            r[:, int(node.emb_scalar)]
            if node.emb_name == "position" and node.emb_scalar is not None and r.ndim == 2
            else r
        )

    def _out_range(
        self, node: NodeInfo, rng: tuple[float, float], is_2d: bool = False
    ) -> tuple[float, float]:
        try:
            if is_2d:
                x = np.linspace(rng[0], rng[1], 50)
                xx, yy = np.meshgrid(x, x)
                inp = np.column_stack([xx.ravel(), yy.ravel()])
            else:
                inp = np.random.uniform(rng[0], rng[1], (1000, 1))
            o = self._eval(node, inp)
            return (float(o.min()), float(o.max()))
        except Exception as e:
            logger.warning(f"Range compute failed for {node.node_type}: {e}")
            return (0.0, 1.0)

    def _compute_ranges(self, groups: dict[str, list[NodeInfo]]):
        self._ranges = {}
        for nt, gk in TOPO_ORDER:
            for n in [x for x in groups.get(gk, []) if _norm(x.node_type) == nt]:
                ir = INV_OUT_RANGE if nt == "inv_translation" else self._in_range(nt)
                self._reg_range(nt, self._out_range(n, ir, is_2d=(nt == "sequestron_ern")))
        logger.info(f"Output ranges: {self._ranges}")

    def _nets(self, units: list[list[list[str]]], src: str | None = None) -> list:
        with LibraryContext.with_library(load_lib()):
            r = Recipe(
                content=[
                    CoTransfection(
                        units=[Unit(slots=s, source=src) if src else Unit(slots=s) for s in u]
                    )
                    for u in units
                ]
            )
            return recipe_to_networks(r, br.ALL_RULES, invert=True, inversion_mode="all")

    def _build_ern(self) -> list[NodeInfo]:
        cc = self.model.compute_config
        if not cc or not cc.node_functions or "sequestron_ERN" not in cc.node_functions:
            return []
        names = [
            n.split("::")[1].split("#")[0]
            for n in cc.node_functions["sequestron_ERN"].kwargs.get("affinity_names", [])
        ]
        if not names:
            return []
        vals = self._emb("shared/quantization/values/affinity", names)
        if not vals:
            logger.info("_build_ern: no affinity embeddings in model params, skipping")
            return []
        nets = [
            n[0]
            for name in names
            if (
                n := self._nets(
                    [
                        [["hEF1a", name], ["hEF1a", "mKO2"]],
                        [["hEF1a", f"{name}_rec", "eYFP"], ["hEF1a", "eBFP2"]],
                    ]
                )
            )
        ]
        if not nets:
            return []
        try:
            st, p, k = self._stack(nets)
        except (KeyError, Exception) as e:
            logger.info(f"_build_ern: stack building failed (no affinity params?): {e}")
            return []
        ly = self._layer(st, "ERN")
        return (
            [
                NodeInfo(
                    name,
                    "ERN",
                    self._apply(ly.f_apply, p, k),
                    i,
                    "affinity",
                    vals.get(name),
                    emb_index=i,
                )
                for i, name in enumerate(names)
            ]
            if ly
            else []
        )

    def _build_translation(self, inv: bool = False) -> list[NodeInfo]:
        cc = self.model.compute_config
        if not cc or not cc.node_functions:
            return []
        try:
            raw = cc.node_functions["translation"].kwargs["quantization_names"]
        except (KeyError, AttributeError):
            return []
        vals = self._emb("shared/quantization/values/tl_rate", raw)
        clean = [(r.replace("_uORF", "") if r != "00_empty_tc" else "none", r) for r in raw]
        nets, nmap = [], {}
        for raw_idx, (c, r) in enumerate(clean):
            if n := self._nets([[["hEF1a", "eBFP2"], ["hEF1a", r, "mKO2"]]]):
                nmap[len(nets)] = (c, r, vals.get(r), raw_idx)
                nets.append(n[0])
        if not nets:
            return []
        try:
            st, p, k = self._stack(nets)
        except (KeyError, Exception) as e:
            logger.warning(f"_build_translation(inv={inv}): stack failed: {e}")
            return []
        lt = "inv_translation" if inv else "translation"
        ly = next((lyr for lyr in st.layers if lyr.type_str() == lt and lyr.f_apply), None)
        if not ly:
            return []
        if inv:
            ns = st.get_layer_namespace(ly.layer_id)
            n2u = {
                v.node_position_in_layer: raw.index(nmap[v.network_id][1])
                for v in ly.nodes
                if v.network_id in nmap
            }
            masks = np.zeros((len(ly.nodes), 1, len(raw)), dtype=bool)
            for nid, uidx in n2u.items():
                masks[nid, :, uidx] = True
            p[f"{ns}/tl_rate_quantization_mask"] = masks
            af = self._apply(ly.f_apply, p, k)
            return [
                NodeInfo(
                    nmap[v.network_id][0],
                    "Inv Translation",
                    af,
                    v.node_position_in_layer,
                    "tl_rate",
                    nmap[v.network_id][2],
                    emb_index=nmap[v.network_id][3],
                )
                for v in ly.nodes
                if v.network_id in nmap
            ]
        af = self._apply(ly.f_apply, p, k)
        res = []
        for ni, (c, r, e, ridx) in nmap.items():
            for v in ly.nodes:
                if v.network_id != ni:
                    continue
                for ed in nets[v.network_id].compute_graph.get_incoming_edges(v.node_id):
                    if ed.content_embedding_names.get("tl_rate") == (r,):
                        cn = [x.name for x in ed.content] if ed.content else []
                        if "mKO2" in cn or r in cn:
                            res.append(
                                NodeInfo(
                                    c,
                                    "Translation",
                                    af,
                                    v.node_position_in_layer,
                                    "tl_rate",
                                    e,
                                    emb_index=ridx,
                                )
                            )
                            break
        return res

    def _build_source(self, inv: bool = False) -> list[NodeInfo]:
        nets = self._nets(BASIC_RECIPE, src="p0")
        if not nets:
            return []
        if inv:
            result = []
            for i, net in enumerate(nets):
                try:
                    st, p, k = self._stack([net])
                except (KeyError, Exception) as e:
                    logger.warning(f"_build_source(inv): stack failed for net {i}: {e}")
                    continue
                ly = next(
                    (lyr for lyr in st.layers if lyr.type_str() == "inv_source" and lyr.f_apply),
                    None,
                )
                if ly:
                    af = self._apply(ly.f_apply, p, k)
                    result.append(NodeInfo(f"pos {i}", "Inv Source", af, 0, "position", float(i)))
            return result
        try:
            st, p, k = self._stack([nets[0]])
        except (KeyError, Exception) as e:
            logger.warning(f"_build_source(fwd): stack failed: {e}")
            return []
        ly = next((lyr for lyr in st.layers if lyr.type_str() == "source" and lyr.f_apply), None)
        if not ly:
            return []
        af = self._apply(ly.f_apply, p, k)
        no = len(ly.f_out_shapes)
        return (
            [NodeInfo(f"pos {i}", "Source", af, 0, "position", float(i)) for i in range(no)]
            if no > 1
            else [NodeInfo(f"pos {i}", "Source", af, i) for i in range(len(ly.nodes))]
        )

    def _build_basic(self) -> tuple[list[NodeInfo], list[NodeInfo], list[NodeInfo]]:
        nets = self._nets(BASIC_RECIPE)
        if not nets:
            return [], [], []
        try:
            st, p, k = self._stack([nets[0]])
        except (KeyError, Exception) as e:
            logger.warning(f"_build_basic: stack building failed: {e}")
            return [], [], []
        fwd, inv, out = [], [], []
        for ly in st.layers:
            if not ly.f_apply:
                continue
            ts = ly.type_str()
            af = self._apply(ly.f_apply, p, k)
            if ts in FWD_LABELS:
                fwd.append(NodeInfo(FWD_LABELS[ts], ts.title(), af, 0))
            if SHOW_INVERSE and ts in INV_LABELS:
                inv.append(NodeInfo(INV_LABELS[ts], ts.replace("_", " ").title(), af, 0))
            if ts == "output":
                out.append(
                    NodeInfo(
                        "PRT → Fluo",
                        "Output",
                        self._apply(ly.f_apply, p, k, n_dup=len(ly.f_input_shapes)),
                        0,
                    )
                )
        return fwd, inv, out

    def _curve(self, ax, node, rng, cmap, col=None, lw=2):
        x = np.linspace(rng[0], rng[1], self.n_trendline_points)
        y = self._eval(node, x.reshape(-1, 1))
        c = col or "black"
        if self.show_distribution:
            np.random.seed(42)
            inp = np.random.uniform(rng[0], rng[1], (self.n_samples, 1))
            ax.scatter(
                inp[:, 0],
                self._eval(node, inp),
                s=4,
                alpha=0.02,
                linewidth=0,
                color=cmap(0.7) if col is None else c,
            )
        ax.plot(x, y, linewidth=lw, color=c)

    def _multi_curve(self, ax, nodes, rng, cmap, title, leg):
        step = max(1, len(nodes) // 8)
        for j, n in enumerate(nodes[::step]):
            col = cmap(j / max(1, len(nodes[::step]) - 1))
            if self.show_distribution:
                np.random.seed(42 + j)
                inp = np.random.uniform(rng[0], rng[1], (self.n_samples, 1))
                ax.scatter(inp[:, 0], self._eval(n, inp), s=4, alpha=0.01, linewidth=0, color=col)
            self._curve(ax, n, rng, cmap, col=col, lw=1.5)
            ax.plot([], [], color=col, label=n.name, linewidth=2)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.legend(title=leg, loc="upper left", fontsize="x-small", frameon=False)
        ax.set(xlabel="Input", ylabel="Output")
        ax.set_box_aspect(1)

    def _emb_heatmap(self, ax, nodes, rng, xlabel, ylabel, title, x_eval_fn=None, show_labels=True):
        sn = sorted(nodes, key=lambda n: n.emb_scalar or 0)
        ev = np.array([n.emb_scalar or 0 for n in sn])
        xr = np.linspace(rng[0], rng[1], 100)
        er = np.linspace(ev.min() - 0.1, ev.max() + 0.1, 100)
        hm = np.zeros((len(er), len(xr)))
        for j, x in enumerate(xr):
            out = np.array(
                [
                    x_eval_fn(n, x)
                    if x_eval_fn
                    else n.apply_fn(x, node_id=n.node_id, random_var=0.5)
                    for n in sn
                ]
            )
            hm[:, j] = np.interp(er, np.sort(ev), out[np.argsort(ev)])
        im = ax.imshow(
            hm,
            extent=[xr[0], xr[-1], er[0], er[-1]],
            origin="lower",
            aspect="auto",
            cmap=self._cmap(False),
        )
        if show_labels:
            for n in sn:
                if n.emb_scalar is not None:
                    ax.axhline(n.emb_scalar, color="black", linestyle="--", alpha=0.7, linewidth=1)
                    ax.text(
                        xr[-1] * 0.5,
                        n.emb_scalar,
                        n.name,
                        color="black",
                        fontsize=9,
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title, fontweight="bold", fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Output")

    @staticmethod
    def _resolve_trajectories_static(
        embedding_trajectories: dict[str, list[tuple[float, ...]]] | None,
        history_dir: Path | None,
        emb_type: str,
        names: list[str],
        name_to_idx: dict[str, int] | None = None,
        max_points: int = 1000,
    ) -> dict[str, list[tuple[float, ...]]]:
        """Resolve trajectories from in-memory data or disk fallback.

        name_to_idx maps node names to their position in the full embedding
        array (codebook index). When None, falls back to positional index
        within ``names``.
        """
        if embedding_trajectories is not None:
            result: dict[str, list[tuple[float, ...]]] = {}
            for i, name in enumerate(names):
                idx = name_to_idx[name] if name_to_idx and name in name_to_idx else i
                key = f"{emb_type}_{idx}"
                if key in embedding_trajectories:
                    result[name] = embedding_trajectories[key]
            return result
        # Fall back to disk loading if history_dir is available
        if history_dir is None or not history_dir.exists():
            return {}
        from biocomptools.logger_history import HistoryManager

        max_steps = max_points
        try:
            batches = HistoryManager.load_from_step_files(history_dir, show_progress=False)
        except Exception as e:
            logger.warning(f"Failed to load step history from {history_dir}: {e}")
            return {}
        if not batches:
            return {}
        if len(batches) > max_steps:
            indices = np.linspace(0, len(batches) - 1, max_steps, dtype=int)
            batches = [batches[i] for i in indices]
        trajectories: dict[str, list[tuple[float, ...]]] = {n: [] for n in names}
        for batch in batches:
            params = batch.arrays.get("latest_params")
            if params is None:
                continue
            try:
                vals_arr = np.asarray(params["shared/quantization/values/" + emb_type])
            except (KeyError, TypeError):
                continue
            while vals_arr.ndim > 2:
                vals_arr = vals_arr[0]
            for i, name in enumerate(names):
                idx = name_to_idx[name] if name_to_idx and name in name_to_idx else i
                if idx < vals_arr.shape[0]:
                    arr = vals_arr[idx].ravel()
                    trajectories[name].append(tuple(float(a) for a in arr))
        return {n: pts for n, pts in trajectories.items() if pts}

    def _resolve_trajectories(
        self, emb_type: str, names: list[str], name_to_idx: dict[str, int] | None = None
    ) -> dict[str, list[tuple[float, ...]]]:
        """Resolve trajectories from in-memory data or disk fallback."""
        return self._resolve_trajectories_static(
            self.embedding_trajectories,
            self.history_dir,
            emb_type,
            names,
            name_to_idx,
            max_points=self.max_trajectory_points,
        )

    def _emb_scatter_2d(self, ax, nodes, emb_type: str, title: str):
        cm = self._cmap(True)
        name_to_idx = {n.name: n.emb_index for n in nodes if n.emb_index is not None} or None
        trajectories = self._resolve_trajectories(emb_type, [n.name for n in nodes], name_to_idx)
        all_coords: list[float] = []
        for i, n in enumerate(nodes):
            if not isinstance(n.emb_val, tuple) or len(n.emb_val) < 2:
                continue
            col = cm(i / max(1, len(nodes) - 1))
            # draw trajectory if available
            if n.name in trajectories and len(trajectories[n.name]) > 1:
                pts = trajectories[n.name]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=col, alpha=0.4, linewidth=1)
                ax.scatter(xs[0], ys[0], color=col, s=20, marker="x", alpha=0.5, zorder=3)
                all_coords.extend(xs)
                all_coords.extend(ys)
            # final position
            ax.scatter(
                n.emb_val[0],
                n.emb_val[1],
                color=col,
                s=50,
                marker="o",
                edgecolors="black",
                linewidth=0.8,
                zorder=5,
            )
            all_coords.append(n.emb_val[0])
            all_coords.append(n.emb_val[1])
            ax.annotate(
                n.name,
                (n.emb_val[0], n.emb_val[1]),
                textcoords="offset points",
                xytext=(5, 0),
                fontsize=8,
                fontweight="bold",
                color=col,
            )

        lo = min(all_coords) if all_coords else -2.0
        hi = max(all_coords) if all_coords else 2.0
        pad = (hi - lo) * 0.1 if hi > lo else 0.5
        lim = (lo - pad, hi + pad)
        ax.set(xlim=lim, ylim=lim)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set(xlabel="Embedding dim 0", ylabel="Embedding dim 1")
        ax.set_box_aspect(1)

    def _emb_visualization(
        self, ax, nodes, emb_type: str, rng, xlabel, ylabel, title, x_eval_fn=None, show_labels=True
    ):
        dim = self._get_emb_dim(emb_type)
        if dim <= 1:
            self._emb_heatmap(ax, nodes, rng, xlabel, ylabel, title, x_eval_fn, show_labels)
        elif dim == 2:
            titles_2d = {
                "tl_rate": "uORF Embedding\n(trajectory)",
                "affinity": "ERN Affinity\n(trajectory)",
            }
            self._emb_scatter_2d(
                ax, nodes, emb_type, titles_2d.get(emb_type, f"{emb_type}\n(trajectory)")
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"Embedding dim={dim}\n(visualization not supported)",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_title(title, fontweight="bold", fontsize=12)
            logger.warning(f"Skipping embedding visualization for {emb_type}: dim={dim} > 2")

    def _ern_2d(self, sf, nodes):
        if not nodes:
            return
        sf.suptitle("ERN Nodes", fontsize=16, fontweight="bold", y=1.05)
        rng = self._in_range("sequestron_ERN")
        x = np.linspace(rng[0], rng[1], 100)
        xx, yy = np.meshgrid(x, x)
        gi = np.column_stack([xx.ravel(), yy.ravel()])
        cm = self._cmap(False)
        ne = min(4, len(nodes))
        axes = sf.subplots(1, ne + 1, width_ratios=[1] * ne + [0.05], gridspec_kw={"wspace": 0.3})
        outs = [self._eval(n, gi) for n in nodes[:ne]]
        vmin, vmax = min(o.min() for o in outs), max(o.max() for o in outs)
        for i, (n, o) in enumerate(zip(nodes[:ne], outs, strict=True)):
            im = axes[i].imshow(
                o.reshape(100, 100),
                extent=[x[0], x[-1], x[0], x[-1]],
                origin="lower",
                aspect="equal",
                cmap=cm,
                vmin=vmin,
                vmax=vmax,
            )
            axes[i].set(xlabel="ERN Protein Amount", ylabel="mRNA Target Amount")
            axes[i].set_title(f"ERN\n({n.name})", fontweight="bold", fontsize=12)
        plt.colorbar(im, cax=axes[-1]).set_label("Output")

    def _ern_1d(self, sf, nodes):
        if not nodes:
            return
        sf.suptitle("ERN Repression Curves", fontsize=16, fontweight="bold", y=1.05)
        axes = sf.subplots(1, 3, width_ratios=[1, 1, 1.1], gridspec_kw={"wspace": 0.3})
        rng = self._in_range("sequestron_ERN")
        nr = np.linspace(rng[0], rng[1], 100)
        span = rng[1] - rng[0]
        ps = [rng[0] + 0.25 * span, rng[0] + 0.75 * span]
        ls = ["-", "--"]
        cm = self._cmap(True)
        cols = {n.name: cm(i / max(1, len(nodes) - 1)) for i, n in enumerate(nodes)}
        for ai, (yl, norm) in enumerate([("Output", False), ("Output / Baseline", True)]):
            ax = axes[ai]
            for n in nodes:
                for j, p in enumerate(ps):
                    out = np.array(
                        [n.apply_fn(x, p, node_id=n.node_id, random_var=0.5) for x in nr]
                    )
                    if norm:
                        out = out / n.apply_fn(0.01, p, node_id=n.node_id, random_var=0.5)
                    ax.plot(
                        nr,
                        out,
                        color=cols[n.name],
                        linewidth=2.5,
                        label=n.name if j == 0 else None,
                        linestyle=ls[j],
                    )
            for j, p in enumerate(ps):
                ax.plot([], [], color="gray", linestyle=ls[j], label=f"pos={p:.2f}")
            if norm:
                ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
                ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
                ax.set(ylim=(-0.05, 1.05))
            ax.set(xlabel="ERN Protein Amount", ylabel=yl)
            ax.set_title(
                "Relative Repression" if norm else "Output vs ERN Protein",
                fontweight="bold",
                fontsize=12,
            )
            ax.legend(fontsize=8, loc="lower left" if norm else "upper right")
            ax.grid(True, alpha=0.3)
        ph = sum(ps) / 2
        self._emb_visualization(
            axes[2],
            nodes,
            "affinity",
            rng,
            "ERN Protein Amount",
            "ERN Embedding",
            f"Output vs ERN Embedding (pos={ph:.2f})",
            x_eval_fn=lambda n, x: n.apply_fn(x, ph, node_id=n.node_id, random_var=0.5),
        )

    def _fwd_row(self, sf, basic, uorf, src, out):
        out = out or []
        if not any([basic, uorf, src, out]):
            return
        sf.suptitle("Forward Nodes", fontsize=16, fontweight="bold", y=1.05)
        cm = self._cmap(True)
        hs, hu = len(src) > 1, bool(uorf)
        np_ = len(basic) + (1 if hs else 0) + (2 if hu else 0) + len(out)
        axes = sf.subplots(1, np_, gridspec_kw={"wspace": 0.3})
        axes = [axes] if np_ == 1 else list(axes)
        ai = 0
        if hs:
            self._multi_curve(
                axes[ai], src, self._in_range("source"), cm, "Source\nplasmid → DNA", "position"
            )
            ai += 1
        for n in basic:
            axes[ai].set_box_aspect(1)
            self._curve(axes[ai], n, self._in_range(n.node_type), cm)
            axes[ai].set(xlabel="Input", ylabel="Output")
            axes[ai].set_title(f"{n.node_type}\n{n.name}", fontweight="bold", fontsize=12)
            ai += 1
        if hu:
            self._multi_curve(
                axes[ai],
                uorf,
                self._in_range("translation"),
                cm,
                "Translation\nmRNA → PRT",
                "uORFs",
            )
            ai += 1
            self._emb_visualization(
                axes[ai],
                uorf,
                "tl_rate",
                self._in_range("translation"),
                "Input (mRNA)",
                "uORF Embedding",
                "Translation\n(input vs uORF emb)",
            )
            ai += 1
        for n in out:
            axes[ai].set_box_aspect(1)
            self._curve(axes[ai], n, self._in_range("output"), cm)
            axes[ai].set(xlabel="Input", ylabel="Output")
            axes[ai].set_title(f"{n.node_type}\n{n.name}", fontweight="bold", fontsize=12)
            ai += 1

    def _inv_row(self, sf, inv, inv_uorf, inv_src):
        inv_src = inv_src or []
        if not inv and not inv_uorf and not inv_src:
            return
        sf.suptitle("Inverse Nodes", fontsize=16, fontweight="bold", y=1.05)
        cm = self._cmap(True)
        hs = len(inv_src) > 1
        np_ = len(inv) + (1 if inv_uorf else 0) + (1 if hs else 0)
        axes = sf.subplots(1, np_, gridspec_kw={"wspace": 0.3})
        axes = [axes] if np_ == 1 else list(axes)
        ai = 0
        if inv_uorf:
            self._multi_curve(
                axes[ai],
                inv_uorf,
                self._in_range("inv_translation"),
                cm,
                "Inv Translation\nPRT → mRNA",
                "uORFs",
            )
            ai += 1
        for n in inv:
            axes[ai].set_box_aspect(1)
            self._curve(axes[ai], n, self._in_range(n.node_type), cm)
            axes[ai].set(xlabel="Input", ylabel="Output")
            axes[ai].set_title(f"{n.node_type}\n{n.name}", fontweight="bold", fontsize=12)
            ai += 1
        if hs:
            self._multi_curve(
                axes[ai],
                inv_src,
                self._in_range("inv_source"),
                cm,
                "Inv Source\nDNA → plasmid",
                "position",
            )
            ai += 1

    def create_figure(self) -> MplFigure:
        ern = sorted(self._build_ern(), key=lambda n: n.emb_scalar or 0, reverse=True)
        uorf = self._build_translation(False)
        inv_uorf = self._build_translation(True)
        src = self._build_source(False)
        inv_src = self._build_source(True)
        basic, inv, out = self._build_basic()
        if uorf:
            basic = [n for n in basic if n.node_type != "Translation"]
        if len(src) > 1:
            basic = [n for n in basic if n.node_type != "Source"]
        if inv_uorf:
            inv = [n for n in inv if n.node_type != "Inv Translation"]
        if len(inv_src) > 1:
            inv = [n for n in inv if n.node_type != "Inv Source"]
        self._compute_ranges(
            {
                "inv_uorf": inv_uorf,
                "inv_src": inv_src,
                "inverse": inv,
                "source": src,
                "basic": basic,
                "uorf": uorf,
                "ern": ern,
                "output": out,
            }
        )

        rows = []
        if ern:
            rows.extend([("ern", ern), ("ern_1d", ern)])
        if any([basic, uorf, src, out]):
            rows.append(("fwd", (basic, uorf, src, out)))
        if (inv or inv_uorf or inv_src) and SHOW_INVERSE:
            rows.append(("inv", (inv, inv_uorf, inv_src)))
        if not rows:
            fig = plt.figure(figsize=(10, 5))
            fig.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=16)
            return fig

        fig = plt.figure(figsize=(20, 5 * len(rows)))
        sfs = fig.subfigures(len(rows), 1, hspace=0.25)
        sfs = [sfs] if len(rows) == 1 else list(sfs)
        fns = {
            "ern": self._ern_2d,
            "ern_1d": self._ern_1d,
            "fwd": lambda s, d: self._fwd_row(s, *d),
            "inv": lambda s, d: self._inv_row(s, *d),
        }
        for sf, (rt, d) in zip(sfs, rows, strict=True):
            try:
                fns[rt](sf, d)
            except Exception as e:
                logger.warning(f"Row '{rt}' rendering failed: {e}")
                sf.text(0.5, 0.5, f"{rt}: render error", ha="center", va="center")
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        return fig

    def run(self, overwrite: bool = True, finalize: bool = True):
        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = self.create_figure()
        fig.savefig(self.figure_spec.output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


class InnerNodesFigureSpec(FigureSpec):
    """Figure spec for inner nodes figure (subclass of FigureSpec for pickling compatibility)."""

    pass
