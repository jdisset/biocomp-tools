"""Markdown-blurb panel renderer.

A lightweight markdown→matplotlib renderer for adding contextual notes
(model description, citation, build info) as a column in `autofig_dataset_row`
layouts. Uses `matplotlib.offsetbox` packers so vertical spacing is handled
by the layout engine — no manual line-height math.

Supported markdown:
  * `# H1`, `## H2`, `### H3`     (heading sizes)
  * `**bold**` / `__bold__`        (inline bold)
  * `*italic*` / `_italic_`        (inline italic)
  * `` `code` ``                   (inline monospace)
  * `[color]text[/color]`         (any matplotlib color name or `#hex`)
  * `- ` / `* ` line prefix        (bullet)
  * blank line                     (paragraph break)

Color tokens nest with inline styles: `[purple]*italic*[/purple]` renders
italic-and-purple. Color names accept `[a-zA-Z0-9_:]+` (so `tab:purple`,
`C0`, etc. work) or `#RGB`/`#RRGGBB`/`#RRGGBBAA`.

Word-wrapping is on by default: each paragraph line is wrapped to fit the
panel's available width while preserving inline styling. Pass
``wrap_chars=int`` to force a column width, or ``wrap=False`` to disable.

Anything richer (links, tables, nested lists, fenced code) falls back to
plain text — the goal is a readable info panel, not a full markdown engine.
"""

from __future__ import annotations

import re
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    HPacker,
    TextArea,
    VPacker,
)


_INLINE_RE = re.compile(
    r"(\*\*[^*]+?\*\*|__[^_]+?__|\*[^*]+?\*|_[^_]+?_|`[^`]+?`)"
)

_COLOR_RE = re.compile(
    r"\[([a-zA-Z][a-zA-Z0-9_:]*|#[0-9a-fA-F]{3,8})\](.+?)\[/\1\]"
)


def _split_color_segments(line: str) -> list[tuple[str, Optional[str]]]:
    """Split a line on `[color]...[/color]` into (text, color_or_None)."""
    out: list[tuple[str, Optional[str]]] = []
    pos = 0
    for m in _COLOR_RE.finditer(line):
        if m.start() > pos:
            out.append((line[pos : m.start()], None))
        out.append((m.group(2), m.group(1)))
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], None))
    return out


def _tokenize_uncolored(line: str) -> list[tuple[str, dict]]:
    """Tokenize bold / italic / code (no color)."""
    out: list[tuple[str, dict]] = []
    pos = 0
    for m in _INLINE_RE.finditer(line):
        if m.start() > pos:
            out.append((line[pos : m.start()], {}))
        tok = m.group(0)
        if tok.startswith("**") or tok.startswith("__"):
            out.append((tok[2:-2], {"fontweight": "bold"}))
        elif tok.startswith("`"):
            out.append((tok[1:-1], {"family": "monospace"}))
        else:  # *italic* / _italic_
            out.append((tok[1:-1], {"style": "italic"}))
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], {}))
    return out


def _tokenize_inline(line: str) -> list[tuple[str, dict]]:
    """Split a line into (text, fontprops) runs, supporting `[color]...[/color]`."""
    out: list[tuple[str, dict]] = []
    for segment, color in _split_color_segments(line):
        for text, props in _tokenize_uncolored(segment):
            merged = dict(props)
            if color is not None:
                merged["color"] = color
            out.append((text, merged))
    return out


def _word_tokens(line: str) -> list[tuple[str, dict]]:
    """Tokenize a line into (word_with_trailing_space, style) chunks suitable
    for greedy word-wrapping. Spaces stay attached to their preceding word so
    line layout doesn't need a separate space-token concept.
    """
    tokens: list[tuple[str, dict]] = []
    for text, style in _tokenize_inline(line):
        for piece in re.findall(r"\S+\s*|\s+", text):
            if piece:
                tokens.append((piece, style))
    return tokens


def _pack_words(words: list[tuple[str, dict]], fontsize: float, extra: Optional[dict] = None):
    """Build an HPacker (or single TextArea) from a list of (text, style) words."""
    base = {"fontsize": fontsize, **(extra or {})}
    if not words:
        return TextArea("", textprops=base)
    if len(words) == 1:
        t, s = words[0]
        return TextArea(t.rstrip(), textprops={**base, **s})
    children = []
    for i, (t, s) in enumerate(words):
        # strip trailing space from the last word in the line so HPacker
        # doesn't render a hanging gap on the right
        text = t.rstrip() if i == len(words) - 1 else t
        children.append(TextArea(text, textprops={**base, **s}))
    return HPacker(children=children, align="baseline", pad=0, sep=0)


def _wrap_words(
    words: list[tuple[str, dict]],
    fontsize: float,
    wrap_chars: int,
    extra: Optional[dict] = None,
) -> list:
    """Greedy wrap into HPacker lines based on character count."""
    if not words:
        return [TextArea("", textprops={"fontsize": fontsize, **(extra or {})})]
    if wrap_chars <= 0:
        return [_pack_words(words, fontsize, extra)]

    lines: list = []
    current: list[tuple[str, dict]] = []
    current_len = 0

    for text, style in words:
        # never start a new line with pure whitespace
        if not current and text.isspace():
            continue
        text_len = len(text)
        if current and current_len + len(text.rstrip()) > wrap_chars:
            lines.append(_pack_words(current, fontsize, extra))
            current = []
            current_len = 0
            if text.isspace():
                continue
        current.append((text, style))
        current_len += text_len
    if current:
        lines.append(_pack_words(current, fontsize, extra))
    return lines


def _estimate_wrap_chars(ax: Axes, fontsize: float, padding_chars: int = 2) -> int:
    """Estimate how many average-width characters fit in the axes' width.
    Uses ~0.55*fontsize per char as a proportional-font approximation."""
    fig = ax.figure
    pos = ax.get_position()
    width_inches = pos.width * fig.get_figwidth()
    char_inches = 0.55 * fontsize / 72.0
    return max(10, int(width_inches / char_inches) - padding_chars)


def render_blurb_to_ax(
    ax: Axes,
    text: str,
    *,
    fontsize: float = 9,
    title: Optional[str] = None,
    title_kwargs: Optional[dict] = None,
    line_sep: float = 3.0,
    block_sep: float = 8.0,
    h1_fontsize: float | None = None,
    h2_fontsize: float | None = None,
    h3_fontsize: float | None = None,
    bullet: str = "•",
    indent: str = "  ",
    wrap: bool = True,
    wrap_chars: int | None = None,
    bullet_indent: int = 2,
    **_kwargs,
):
    """Render a markdown blurb into ``ax`` as a left-aligned, top-anchored block.

    The axes are stripped of ticks and spines; the layout is handled by
    matplotlib's offsetbox packers, so there's no fragile line-height math.

    Word-wrapping is on by default — paragraph and bullet lines wrap to fit
    the panel's width while preserving inline styling. ``wrap_chars`` lets
    you force a fixed column width; ``wrap=False`` disables wrapping.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, **(title_kwargs or {}))

    if not text:
        return

    h1 = h1_fontsize or fontsize + 4
    h2 = h2_fontsize or fontsize + 2
    h3 = h3_fontsize or fontsize + 1

    if wrap:
        body_chars = wrap_chars if wrap_chars is not None else _estimate_wrap_chars(ax, fontsize)
        bullet_chars = max(10, body_chars - bullet_indent)
        h1_chars = wrap_chars if wrap_chars is not None else _estimate_wrap_chars(ax, h1)
        h2_chars = wrap_chars if wrap_chars is not None else _estimate_wrap_chars(ax, h2)
        h3_chars = wrap_chars if wrap_chars is not None else _estimate_wrap_chars(ax, h3)
    else:
        body_chars = bullet_chars = h1_chars = h2_chars = h3_chars = 0  # 0 -> no wrap

    def wrapped_block(line: str, fs: float, chars: int, extra: Optional[dict] = None):
        words = _word_tokens(line)
        lines = _wrap_words(words, fs, chars, extra)
        if len(lines) == 1:
            return lines[0]
        return VPacker(children=lines, align="left", pad=0, sep=line_sep * 0.5)

    blocks: list = []           # outer VPacker children: paragraphs
    current: list = []          # inner VPacker children: lines in current paragraph

    def flush():
        if current:
            blocks.append(VPacker(children=list(current), align="left", pad=0, sep=line_sep))
            current.clear()

    for raw in text.strip().splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if line.startswith("### "):
            flush()
            blocks.append(wrapped_block(line[4:], h3, h3_chars, {"fontweight": "bold"}))
        elif line.startswith("## "):
            flush()
            blocks.append(wrapped_block(line[3:], h2, h2_chars, {"fontweight": "bold"}))
        elif line.startswith("# "):
            flush()
            blocks.append(wrapped_block(line[2:], h1, h1_chars, {"fontweight": "bold"}))
        elif line.startswith(("- ", "* ")):
            inner = wrapped_block(line[2:], fontsize, bullet_chars)
            bullet_pack = HPacker(
                children=[TextArea(f"{bullet} ", textprops={"fontsize": fontsize}), inner],
                align="top",
                pad=0,
                sep=0,
            )
            current.append(bullet_pack)
        elif line.startswith(indent):
            current.append(wrapped_block(line.lstrip(), fontsize, body_chars))
        else:
            current.append(wrapped_block(line, fontsize, body_chars))
    flush()

    outer = VPacker(children=blocks, align="left", pad=0, sep=block_sep)
    abox = AnchoredOffsetbox(
        loc="upper left",
        child=outer,
        frameon=False,
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax.add_artist(abox)


__all__ = ["render_blurb_to_ax"]
