"""One-shot migration: fill `input_order` on every recipe missing it.

Walks every `*.recipe.yaml` under `$BIOCOMP_ROOT/Experiments/`, parses each
through Dracon (handling the optional `_metadata:` preamble), and writes back
a non-empty `input_order` field derived from the SSOT helper in
`biocomp.recipe.default_input_order_for_network`. Idempotent: re-running on
already-migrated recipes is a no-op.

Usage:
    uv run python biocomp-tools/scripts/migrate_recipe_input_order.py --dry-run
    uv run python biocomp-tools/scripts/migrate_recipe_input_order.py --apply

After --apply, run `biocomp-updatedb` to refresh the SQLite `Recipe.content`
rows from the migrated YAMLs.

See `bugs/eval-x-axis-permutation-fix-plan.md` Phase 1 for context.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from biocomp.library import load_lib
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe, default_input_order_for_network


_RECIPE_TAG = "!biocomp.recipe.Recipe"


@dataclass
class MigrationOutcome:
    path: Path
    status: str  # "filled", "skipped_already_set", "skipped_no_inputs", "failed"
    proposed: Optional[list[str]] = None
    reason: Optional[str] = None


def _split_metadata_preamble(text: str) -> tuple[str, str]:
    """Split optional `_metadata:` preamble (used by design pipelines) from the
    `!biocomp.recipe.Recipe` body. Returns (preamble, recipe_body)."""
    marker = f"\n{_RECIPE_TAG}"
    if marker in text:
        idx = text.index(marker)
        return text[: idx + 1], text[idx + 1 :]
    return "", text


def _load_recipe(path: Path) -> Recipe:
    """Load a Recipe from disk, accommodating the optional metadata preamble."""
    return Recipe.load_from_paper_yaml(path)


def _propose_input_order(recipe: Recipe, lib) -> list[str]:
    """Use SSOT helper to compute the default input order for this recipe.

    Builds inverted networks once, then asks the helper. Fails loudly when
    different networks would yield different proposals — that's a recipe-author
    bug to resolve manually rather than something migration can guess at.
    """
    nets = recipe_to_networks(recipe, lib=lib, invert=True)
    proposals: list[list[str]] = []
    for net in nets:
        proteins = net.get_inverted_input_proteins()
        if not proteins:
            continue
        proposals.append(default_input_order_for_network(net))

    if not proposals:
        raise ValueError("recipe yields no networks with inputs — cannot migrate")

    first = proposals[0]
    for p in proposals[1:]:
        if p != first:
            raise ValueError(
                f"recipe yields >1 network with different input proposals: "
                f"{proposals}. Resolve manually."
            )
    return first


def _write_input_order(path: Path, input_order: list[str]) -> None:
    """Insert `input_order: [...]` into the YAML body, preserving the
    `_metadata:` preamble. Inserts immediately after the `!biocomp.recipe.Recipe`
    line so the field placement is consistent and dracon-loadable.
    """
    text = path.read_text()
    preamble, body = _split_metadata_preamble(text)
    lines = body.splitlines(keepends=True)

    out_lines: list[str] = []
    inserted = False
    for line in lines:
        out_lines.append(line)
        if not inserted and line.rstrip().startswith(_RECIPE_TAG):
            indent = ""  # top-level field
            rendered = ", ".join(repr(p) for p in input_order)
            out_lines.append(f"{indent}input_order: [{rendered}]\n")
            inserted = True

    if not inserted:
        raise RuntimeError(f"could not locate `{_RECIPE_TAG}` line in {path}")

    path.write_text(preamble + "".join(out_lines))


def migrate_one(path: Path, lib, *, apply: bool) -> MigrationOutcome:
    try:
        recipe = _load_recipe(path)
    except Exception as e:
        return MigrationOutcome(path, "failed", reason=f"load error: {e}")

    if recipe.input_order:
        return MigrationOutcome(path, "skipped_already_set")

    try:
        proposed = _propose_input_order(recipe, lib)
    except ValueError as e:
        msg = str(e)
        if "no networks with inputs" in msg:
            return MigrationOutcome(path, "skipped_no_inputs", reason=msg)
        return MigrationOutcome(path, "failed", reason=msg)
    except Exception as e:
        return MigrationOutcome(path, "failed", reason=f"propose error: {e}")

    if apply:
        try:
            _write_input_order(path, proposed)
        except Exception as e:
            return MigrationOutcome(path, "failed", proposed=proposed, reason=f"write error: {e}")

    return MigrationOutcome(path, "filled", proposed=proposed)


def discover_recipes(root: Path) -> list[Path]:
    return sorted(root.rglob("*.recipe.yaml"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only; do not write files")
    parser.add_argument("--apply", action="store_true", help="write filled input_order to disk")
    parser.add_argument(
        "--root",
        default=None,
        help="recipe root (defaults to $BIOCOMP_ROOT/Experiments)",
    )
    args = parser.parse_args()

    if args.apply == args.dry_run:
        parser.error("specify exactly one of --dry-run or --apply")

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        biocomp_root = os.environ.get("BIOCOMP_ROOT")
        if not biocomp_root:
            parser.error("BIOCOMP_ROOT must be set (or pass --root)")
        root = Path(biocomp_root) / "Experiments"

    if not root.is_dir():
        parser.error(f"recipe root does not exist: {root}")

    paths = discover_recipes(root)
    print(f"found {len(paths)} recipe files under {root}")

    lib = load_lib()

    counts: dict[str, int] = {}
    failed: list[MigrationOutcome] = []
    for p in paths:
        outcome = migrate_one(p, lib, apply=args.apply)
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
        if outcome.status == "failed":
            failed.append(outcome)
        if outcome.status == "filled":
            print(
                f"{'WROTE' if args.apply else 'WOULD WRITE'}: {p}  "
                f"input_order={outcome.proposed}"
            )

    print("\nsummary:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    if failed:
        print("\nfailures:")
        for f in failed:
            print(f"  {f.path}: {f.reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
