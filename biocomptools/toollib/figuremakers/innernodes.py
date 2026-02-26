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
INV_TL_RANGE = (-0.1, 0.7)
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
    emb_val: float | None = None


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

    _cache: dict = {}
    _ranges: dict[str, tuple[float, float]] = {}

    def _cmap(self, truncate: bool = True):
        base = plt.get_cmap(CMAP)
        return LinearSegmentedColormap.from_list(
            "c", base(np.linspace(CMAP_TRUNCATE_MIN if truncate else 0, 1, 256))
        )

    def _emb(self, path: str, names: list[str]) -> dict[str, float]:
        try:
            v: Any = self.model.shared_params
            for k in path.split("/"):
                v = v[k]
            return {n: float(x[0]) for n, x in zip(names, v, strict=True)}
        except (KeyError, IndexError, TypeError):
            return {}

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
            return INV_TL_RANGE
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
            r[:, int(node.emb_val)]
            if node.emb_name == "position" and node.emb_val is not None and r.ndim == 2
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
                ir = INV_TL_RANGE if nt == "inv_translation" else self._in_range(nt)
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
        st, p, k = self._stack(nets)
        ly = self._layer(st, "ERN")
        return (
            [
                NodeInfo(name, "ERN", self._apply(ly.f_apply, p, k), i, "affinity", vals.get(name))
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
        for c, r in clean:
            if n := self._nets([[["hEF1a", "eBFP2"], ["hEF1a", r, "mKO2"]]]):
                nmap[len(nets)] = (c, r, vals.get(r))
                nets.append(n[0])
        if not nets:
            return []
        st, p, k = self._stack(nets)
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
                )
                for v in ly.nodes
                if v.network_id in nmap
            ]
        af = self._apply(ly.f_apply, p, k)
        res = []
        for ni, (c, r, e) in nmap.items():
            for v in ly.nodes:
                if v.network_id != ni:
                    continue
                for ed in nets[v.network_id].compute_graph.get_incoming_edges(v.node_id):
                    if ed.content_embedding_names.get("tl_rate") == (r,):
                        cn = [x.name for x in ed.content] if ed.content else []
                        if "mKO2" in cn or r in cn:
                            res.append(
                                NodeInfo(
                                    c, "Translation", af, v.node_position_in_layer, "tl_rate", e
                                )
                            )
                            break
        return res

    def _build_source(self, inv: bool = False) -> list[NodeInfo]:
        nets = self._nets(BASIC_RECIPE, src="p0")
        if not nets:
            return []
        if inv:
            # For inverse source, each network inversion gives us a different position
            # Use all networks to get inv_source for each position
            result = []
            for i, net in enumerate(nets):
                st, p, k = self._stack([net])
                ly = next(
                    (lyr for lyr in st.layers if lyr.type_str() == "inv_source" and lyr.f_apply),
                    None,
                )
                if ly:
                    af = self._apply(ly.f_apply, p, k)
                    result.append(NodeInfo(f"pos {i}", "Inv Source", af, 0, "position", float(i)))
            return result
        # Forward source - single network, multiple output positions
        st, p, k = self._stack([nets[0]])
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
        st, p, k = self._stack([nets[0]])
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
        sn = sorted(nodes, key=lambda n: n.emb_val or 0)
        ev = np.array([n.emb_val or 0 for n in sn])
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
                if n.emb_val is not None:
                    ax.axhline(n.emb_val, color="black", linestyle="--", alpha=0.7, linewidth=1)
                    ax.text(
                        xr[-1] * 0.5,
                        n.emb_val,
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
        self._emb_heatmap(
            axes[2],
            nodes,
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
            self._emb_heatmap(
                axes[ai],
                uorf,
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
                axes[ai], inv_uorf, self._in_range("inv_translation"), cm, "Inv Translation\nPRT → mRNA", "uORFs"
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
        ern = sorted(self._build_ern(), key=lambda n: n.emb_val or 0, reverse=True)
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
            fns[rt](sf, d)
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
