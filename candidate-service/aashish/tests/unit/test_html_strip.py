from __future__ import annotations

from src.utils.html import strip_html


def test_strip_html_handles_empty():
    assert strip_html("") == ""
    assert strip_html(None) == ""


def test_strip_html_paragraph_and_list():
    html = (
        "<h3>About</h3><p>We build things.</p>"
        "<ul><li>One</li><li>Two</li></ul>"
    )
    out = strip_html(html)
    assert "About" in out
    assert "We build things." in out
    assert "One" in out and "Two" in out
    assert "<" not in out and ">" not in out


def test_strip_html_collapses_whitespace_within_lines():
    html = "<p>hello    world</p>"
    out = strip_html(html)
    assert out == "hello world"


def test_strip_html_preserves_line_separation():
    html = "<p>alpha</p><p>beta</p>"
    out = strip_html(html)
    assert "alpha" in out and "beta" in out
    assert "\n" in out
