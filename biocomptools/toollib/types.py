# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Type definitions for biocomptools.toollib

Single source of truth for common types used across datasources, networkprediction, etc.
"""

from typing import Literal, Sequence, Union

InputOrderElement = Union[int, str]
"""Single element in an input order specification: index (int) or protein name (str)."""

InputOrderSpec = Union[Sequence[InputOrderElement], Literal["inv"]]
"""Input order specification: sequence of indices/names, or "inv" for reverse alphabetical."""
