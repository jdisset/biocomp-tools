# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Helpers for sample efficiency studies.

Used by broodmon YAML jobs to generate deterministic random subsets
of uORF matrix combinations for training experiments.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
from pydantic import BaseModel


class PguSubsetGenerator(BaseModel):
    """Generate deterministic random PgU matrix subsets for sample efficiency studies.

    Each (k, replicate_id) pair maps to a unique random subset of k uORF pairs
    from the full 9x9 PgU matrix, seeded as k * 1000 + replicate_id.

    Properties:
      - ``jobs``: random-subset jobs for sample_efficiency_pgu broodmon job
      - ``named_scenario_jobs``: named scenarios for named_matrix_scenarios broodmon job

    Usage from broodmon YAML:
        !noconstruct _gen: !biocomptools.toollib.sample_efficiency.PguSubsetGenerator
          sample_counts: ${sample_counts}
          n_replicates: ${n_replicates}
        !define jobs: ${construct(&/_gen).jobs}
    """

    available: list[int] = [0, 5, 10, 20, 30, 40, 50, 60, 80]
    sample_counts: list[int] = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64]
    n_replicates: int = 10

    @cached_property
    def full_matrix(self) -> list[tuple[int, int]]:
        return [(x, y) for x in self.available for y in self.available]

    @cached_property
    def named_scenarios(self) -> dict[str, list[list[int]]]:
        """Named matrix subsets, mirroring biocomp-jobs/train/matrices/var_definitions.yaml."""
        a = self.available
        return {
            "single": [[0, 0]],
            "single_8x8": [[80, 80]],
            "3corners": [[0, 0], [0, 80], [80, 0]],
            "4corners": [[0, 0], [0, 80], [80, 0], [80, 80]],
            "diagonal": [[x, x] for x in a],
            "inner_box1": [
                [20, 20], [20, 30], [30, 20], [40, 40],
                [40, 30], [30, 40], [20, 40], [40, 20],
            ],
            "inner_4corners": [[20, 20], [20, 40], [40, 20], [40, 40]],
            "bounding_box1": [
                list(p) for p in dict.fromkeys(
                    tuple(p)
                    for p in [[x, y] for x in [0, 80] for y in a]
                    + [[x, y] for x in a for y in [0, 80]]
                )
            ],
            "full_matrix": [[x, y] for x in a for y in a],
        }

    def generate_subset(self, k: int, replicate_id: int) -> list[list[int]]:
        """Deterministic random subset of k uORF pairs."""
        rng = np.random.default_rng(seed=k * 1000 + replicate_id)
        indices = rng.choice(len(self.full_matrix), size=k, replace=False)
        return [list(self.full_matrix[i]) for i in sorted(indices)]

    @cached_property
    def named_scenario_jobs(self) -> list[dict[str, object]]:
        """Job dicts for named scenarios, for use by named_matrix_scenarios broodmon job."""
        return [
            {"name": name, "k": len(subset), "subset": str(subset).replace(" ", "")}
            for name, subset in self.named_scenarios.items()
        ]

    @cached_property
    def jobs(self) -> list[dict[str, object]]:
        """Random-subset jobs, shuffled with fixed seed for balanced load.

        Deterministic shuffle ensures consecutive jobs have mixed k values,
        so any sliding window of parallel workers sees balanced work.
        """
        result: list[dict[str, object]] = []
        for k in self.sample_counts:
            for r in range(self.n_replicates):
                subset = self.generate_subset(k, r)
                result.append(
                    {
                        "k": k,
                        "r": r,
                        "subset": str(subset).replace(" ", ""),
                    }
                )
        np.random.default_rng(seed=42).shuffle(result)  # pyright: ignore[reportArgumentType]
        return result
