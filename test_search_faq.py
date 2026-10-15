"""Basic ranking test for the GreenField Farmers FAQ search."""
from search_faq import search_faq


def test_membership_query_returns_expected_faq_first():
    results = search_faq("How do I become a member of the farmers cooperative?")

    assert len(results) == 3
    assert results[0]["question"] == "How can I become a member of GreenField Farmers?"
    assert all("question" in item and "answer" in item for item in results)
    assert results[0]["similarity_score"] >= results[1]["similarity_score"]
    assert results[1]["similarity_score"] >= results[2]["similarity_score"]
