import pytest

from hdb_rag.discovery.filters import is_junk_url


@pytest.mark.parametrize("url", [
    "https://www.hdb.gov.sg/residential/buying-a-flat/financing/eligibility",  # legit
    "https://www.hdb.gov.sg/residential/living-in-an-hdb-flat/sub-letting",
    "https://www.hdb.gov.sg/cs/Satellite?c=Resource&pagename=...&pdf=BTO.pdf",  # PDF
])
def test_legit_urls_are_not_junk(url):
    assert is_junk_url(url) is False


@pytest.mark.parametrize("url", [
    "https://www.hdb.gov.sg/cs/sitemap",
    "https://www.hdb.gov.sg/cs/contact-us",
    "https://www.hdb.gov.sg/cs/privacy-statement",
    "https://www.hdb.gov.sg/cs/login",
    "https://www.hdb.gov.sg/cs/about-us/news-room",
    "https://www.hdb.gov.sg/cs/search?q=BTO",
    "https://www.hdb.gov.sg/residential/buying-a-flat/eligibility#section-2",  # anchor
    "javascript:void(0)",
    "mailto:contact@hdb.gov.sg",
])
def test_junk_urls_are_filtered(url):
    assert is_junk_url(url) is True
