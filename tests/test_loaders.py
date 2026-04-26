from pathlib import Path

from hdb_rag.ingest.loaders import _clean_html, _next_data_text, load_html_file


def test_html_loader_extracts_main_text():
    fixture = Path(__file__).parent / "fixtures" / "sample_page.html"
    text = load_html_file(fixture)
    assert "Singapore Citizen aged 21" in text
    assert "FOOTER STUFF" not in text
    assert "NAV STUFF" not in text


def test_next_data_html_fragments_preserve_line_breaks():
    html = """
    <html><body>
      <script id="__NEXT_DATA__" type="application/json">
        {"props": {"pageProps": {"bodyContentVal": "<h2>Eligibility</h2><p>Apply if eligible.</p>"}}}
      </script>
    </body></html>
    """

    text = _next_data_text(html)

    assert "Eligibility\nApply if eligible." in text


def test_html_loader_dedupes_visible_next_data_boundary_heading():
    html = """
    <html><body>
      <main><h1>Eligibility</h1></main>
      <script id="__NEXT_DATA__" type="application/json">
        {"props": {"pageProps": {"bodyContentVal": "<h1>Eligibility</h1><p>Apply if eligible.</p>"}}}
      </script>
    </body></html>
    """

    text = _clean_html(html)
    lines = [line for line in text.splitlines() if line.strip()]

    assert lines == ["Eligibility", "Apply if eligible."]


def test_html_loader_dedupes_repeated_initial_visible_heading():
    html = """
    <html><body>
      <main>
        <h1>Eligibility</h1>
        <div>Eligibility</div>
        <p>Apply if eligible.</p>
      </main>
    </body></html>
    """

    text = _clean_html(html)
    lines = [line for line in text.splitlines() if line.strip()]

    assert lines == ["Eligibility", "Apply if eligible."]


def test_html_loader_preserves_repeated_table_like_lines_inside_content():
    html = """
    <html><body>
      <script id="__NEXT_DATA__" type="application/json">
        {"props": {"pageProps": {"bodyContentVal": "<p>$40,000</p><p>$40,000</p><p>5%</p><p>5%</p>"}}}
      </script>
    </body></html>
    """

    text = _clean_html(html)
    lines = [line for line in text.splitlines() if line.strip()]

    assert lines == ["$40,000", "$40,000", "5%", "5%"]
