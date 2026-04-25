import yaml

from hdb_rag.discovery.emitter import emit_sources_yaml


def test_emit_sorts_and_serializes(tmp_path):
    pages = [
        {"url": "https://www.hdb.gov.sg/b", "title": "B", "doc_type": "html", "category": "buying"},
        {"url": "https://www.hdb.gov.sg/a", "title": "A", "doc_type": "html", "category": "buying"},
    ]
    out_path = tmp_path / "sources.yaml"
    emit_sources_yaml(pages, out_path)

    data = yaml.safe_load(out_path.read_text())
    urls = [s["url"] for s in data["sources"]]
    assert urls == ["https://www.hdb.gov.sg/a", "https://www.hdb.gov.sg/b"]
    assert data["sources"][0]["title"] == "A"
    assert data["sources"][0]["category"] == "buying"
    assert data["sources"][0]["type"] == "html"
