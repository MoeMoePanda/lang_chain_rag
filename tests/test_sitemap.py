import responses

from hdb_rag.discovery.sitemap import discover_from_sitemap


SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.hdb.gov.sg/</loc></url>
  <url><loc>https://www.hdb.gov.sg/about-us</loc></url>
  <url><loc>https://www.hdb.gov.sg/buying-a-flat/eligibility</loc></url>
  <url><loc>https://www.hdb.gov.sg/buying-a-flat/bto-sbf-and-open-booking-of-flats/finding-a-new-flat/types-of-flats</loc></url>
  <url><loc>https://www.hdb.gov.sg/managing-my-home/finances/loan-matters/refinance</loc></url>
  <url><loc>https://www.hdb.gov.sg/about-us/contact-us</loc></url>
  <url><loc>https://www.hdb.gov.sg/hdb-pulse/news/whats-on</loc></url>
  <url><loc>https://www.hdb.gov.sg/buying-a-flat/sample.pdf</loc></url>
</urlset>
"""

ROBOTS = "User-agent: *\nAllow: /\n"


@responses.activate
def test_sitemap_filters_by_scope_and_drops_junk():
    responses.add(responses.GET, "https://www.hdb.gov.sg/robots.txt", body=ROBOTS, status=200)
    responses.add(
        responses.GET,
        "https://www.hdb.gov.sg/sitemap.xml",
        body=SITEMAP_XML,
        status=200,
        content_type="application/xml",
    )

    pages = discover_from_sitemap(
        sitemap_url="https://www.hdb.gov.sg/sitemap.xml",
        scope_paths=["/buying-a-flat/", "/managing-my-home/"],
        user_agent="test-agent",
        request_delay=0.0,
    )

    urls = {p["url"] for p in pages}
    assert "https://www.hdb.gov.sg/buying-a-flat/eligibility" in urls
    assert "https://www.hdb.gov.sg/buying-a-flat/bto-sbf-and-open-booking-of-flats/finding-a-new-flat/types-of-flats" in urls
    assert "https://www.hdb.gov.sg/managing-my-home/finances/loan-matters/refinance" in urls
    assert "https://www.hdb.gov.sg/buying-a-flat/sample.pdf" in urls

    # out of scope or junk
    assert "https://www.hdb.gov.sg/" not in urls
    assert "https://www.hdb.gov.sg/about-us" not in urls
    assert "https://www.hdb.gov.sg/about-us/contact-us" not in urls
    assert "https://www.hdb.gov.sg/hdb-pulse/news/whats-on" not in urls


@responses.activate
def test_sitemap_assigns_categories_and_doc_types():
    responses.add(responses.GET, "https://www.hdb.gov.sg/robots.txt", body=ROBOTS, status=200)
    responses.add(
        responses.GET,
        "https://www.hdb.gov.sg/sitemap.xml",
        body=SITEMAP_XML,
        status=200,
        content_type="application/xml",
    )
    pages = discover_from_sitemap(
        sitemap_url="https://www.hdb.gov.sg/sitemap.xml",
        scope_paths=["/buying-a-flat/", "/managing-my-home/"],
        user_agent="test-agent",
        request_delay=0.0,
    )
    by_url = {p["url"]: p for p in pages}

    elig = by_url["https://www.hdb.gov.sg/buying-a-flat/eligibility"]
    assert elig["category"] == "buying"
    assert elig["doc_type"] == "html"
    assert elig["title"]  # derived; non-empty

    refinance = by_url["https://www.hdb.gov.sg/managing-my-home/finances/loan-matters/refinance"]
    assert refinance["category"] == "living"

    pdf = by_url["https://www.hdb.gov.sg/buying-a-flat/sample.pdf"]
    assert pdf["doc_type"] == "pdf"
