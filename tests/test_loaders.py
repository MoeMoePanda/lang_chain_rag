from pathlib import Path

from hdb_rag.ingest.loaders import load_html_file


def test_html_loader_extracts_main_text():
    fixture = Path(__file__).parent / "fixtures" / "sample_page.html"
    text = load_html_file(fixture)
    assert "Singapore Citizen aged 21" in text
    assert "FOOTER STUFF" not in text
    assert "NAV STUFF" not in text
