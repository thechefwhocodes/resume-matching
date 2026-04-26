"""HTML stripping for job responsibilities so embeddings + BM25 see clean text."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

_WHITESPACE = re.compile(r"\s+")


def strip_html(html: str | None) -> str:
    """Convert an HTML fragment to plain text.

    Uses BeautifulSoup with the built-in html.parser so we don't pull in lxml.
    Block-level elements (h1-h6, p, li, br) become newline-separated paragraphs;
    multiple whitespace runs collapse to a single space within each paragraph.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
