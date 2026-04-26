from pathlib import Path

from hdb_rag.ingest.loaders import _next_data_text, load_html_file


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
