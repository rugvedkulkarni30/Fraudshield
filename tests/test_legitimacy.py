"""
test_legitimacy.py — Tests for external legitimacy verification module.
========================================================================
Includes unit tests with mocked network calls and one live integration test.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.legitimacy import (
    ATS_DOMAINS,
    FREE_EMAIL_DOMAINS,
    batch_check,
    check_posting,
    extract_registered_domain,
)


class TestDomainExtractionAndRules:
    def test_extract_registered_domain(self):
        assert extract_registered_domain("user@careers.google.com") == "google.com"
        assert extract_registered_domain("https://jobs.lever.co/acme/apply") == "lever.co"
        assert extract_registered_domain("http://www.example.co.uk:8080/path") == "example.co.uk"
        assert extract_registered_domain(None) is None
        assert extract_registered_domain("") is None
        assert extract_registered_domain("nan") is None

    def test_unknown_values_are_none_never_zero_or_false(self):
        """When fields are missing or cannot be resolved, values must be None, not False or 0."""
        result = check_posting(contact_email=None, application_url=None, company_website=None)

        assert result["contact_domain"] is None
        assert result["company_domain"] is None
        assert result["domain_age_days"] is None
        assert result["domain_match"] is None
        assert result["free_email"] is None
        assert result["is_ats_domain"] is None
        assert result["mx_present"] is None
        assert result["https_valid"] is None
        assert result["cert_days_left"] is None

        # Verify none of them fell back to 0 or False
        for key in [
            "domain_age_days",
            "domain_match",
            "free_email",
            "is_ats_domain",
            "mx_present",
            "https_valid",
            "cert_days_left",
        ]:
            assert result[key] is not False, f"{key} should not be False"
            assert result[key] != 0 or result[key] is None, f"{key} should not be 0"

    def test_privacy_preservation(self):
        """Verify email local-parts and any potential PII are never returned in results."""
        sensitive_email = "very_secret_username_42@example.com"
        with patch("src.legitimacy.check_mx_records", return_value=True), \
             patch("src.legitimacy.fetch_domain_creation_date", return_value="2020-01-01"), \
             patch("src.legitimacy.check_https_and_cert", return_value=(True, 90, "2027-01-01T00:00:00+00:00")):
            result = check_posting(
                contact_email=sensitive_email,
                application_url="https://example.com/apply",
                company_website="https://example.com",
            )

        assert "very_secret_username_42" not in str(result)
        assert result["contact_domain"] == "example.com"
        # Ensure no registrant fields exist in dict
        for forbidden in ["registrant", "name", "address", "phone", "admin", "tech"]:
            assert forbidden not in result

    def test_free_email_detection(self):
        with patch("src.legitimacy.check_mx_records", return_value=True), \
             patch("src.legitimacy.fetch_domain_creation_date", return_value="2000-01-01"), \
             patch("src.legitimacy.check_https_and_cert", return_value=(True, 120, "2027-01-01T00:00:00+00:00")):
            res_free = check_posting(contact_email="recruiter@gmail.com")
            assert res_free["free_email"] is True
            assert res_free["contact_domain"] == "gmail.com"

            res_corp = check_posting(contact_email="recruiter@microsoft.com")
            assert res_corp["free_email"] is False
            assert res_corp["contact_domain"] == "microsoft.com"

    def test_ats_detection(self):
        with patch("src.legitimacy.check_mx_records", return_value=True), \
             patch("src.legitimacy.fetch_domain_creation_date", return_value="2010-01-01"), \
             patch("src.legitimacy.check_https_and_cert", return_value=(True, 90, "2027-01-01T00:00:00+00:00")):
            res_ats = check_posting(
                contact_email="hr@mycorp.com",
                application_url="https://jobs.lever.co/mycorp/123",
                company_website="https://mycorp.com",
            )
            assert res_ats["is_ats_domain"] is True

            res_non_ats = check_posting(
                contact_email="hr@mycorp.com",
                application_url="https://mycorp.com/careers",
                company_website="https://mycorp.com",
            )
            assert res_non_ats["is_ats_domain"] is False

    def test_domain_match_logic(self):
        with patch("src.legitimacy.check_mx_records", return_value=True), \
             patch("src.legitimacy.fetch_domain_creation_date", return_value="2015-01-01"), \
             patch("src.legitimacy.check_https_and_cert", return_value=(True, 90, "2027-01-01T00:00:00+00:00")):
            # Matching
            res_match = check_posting(
                contact_email="jobs@acme.org",
                company_website="https://acme.org",
            )
            assert res_match["domain_match"] is True

            # Mismatched
            res_mismatch = check_posting(
                contact_email="jobs@acme-scam.org",
                company_website="https://acme.org",
            )
            assert res_mismatch["domain_match"] is False

            # Missing one
            res_partial = check_posting(
                contact_email="jobs@acme.org",
                company_website=None,
            )
            assert res_partial["domain_match"] is None


class TestMockedNetworkAndCache:
    @patch("src.legitimacy.check_mx_records", return_value=True)
    @patch("src.legitimacy.fetch_domain_creation_date", return_value="2010-06-15")
    @patch("src.legitimacy.check_https_and_cert", return_value=(True, 150, "2027-01-01T00:00:00+00:00"))
    def test_mocked_network_lookups(self, mock_https, mock_age, mock_mx, tmp_path):
        # Override cache DB to tmp_path
        with patch("src.legitimacy.CACHE_DB_PATH", tmp_path / "test_cache.sqlite"), \
             patch("src.legitimacy.CACHE_DIR", tmp_path):
            res = check_posting(
                contact_email="careers@example.com",
                application_url="https://example.com/jobs",
                company_website="https://example.com",
            )

            assert res["contact_domain"] == "example.com"
            assert res["company_domain"] == "example.com"
            assert res["domain_match"] is True
            assert res["mx_present"] is True
            assert res["https_valid"] is True
            assert res["cert_days_left"] == 150
            assert res["domain_age_days"] is not None
            assert res["domain_age_days"] > 3000

    @patch("src.legitimacy.check_mx_records", return_value=True)
    @patch("src.legitimacy.fetch_domain_creation_date", return_value="2015-01-01")
    @patch("src.legitimacy.check_https_and_cert", return_value=(True, 200, "2027-01-01T00:00:00+00:00"))
    def test_sqlite_cache_prevents_repeated_calls(self, mock_https, mock_age, mock_mx, tmp_path):
        cache_db = tmp_path / "test_cache.sqlite"
        with patch("src.legitimacy.CACHE_DB_PATH", cache_db), \
             patch("src.legitimacy.CACHE_DIR", tmp_path):
            # First call triggers network mocks
            res1 = check_posting(company_website="https://cached-example.com")
            assert mock_age.call_count == 1
            assert mock_https.call_count == 1

            # Second call should read from SQLite cache
            res2 = check_posting(company_website="https://cached-example.com")
            assert mock_age.call_count == 1  # Not called again
            assert mock_https.call_count == 1  # Not called again
            assert res1["domain_age_days"] == res2["domain_age_days"]
            assert res1["https_valid"] == res2["https_valid"]


class TestBatchCheck:
    @patch("src.legitimacy.check_mx_records", return_value=True)
    @patch("src.legitimacy.fetch_domain_creation_date", return_value="2018-01-01")
    @patch("src.legitimacy.check_https_and_cert", return_value=(True, 60, "2027-01-01T00:00:00+00:00"))
    def test_batch_check_dataframe(self, mock_https, mock_age, mock_mx, tmp_path):
        df = pd.DataFrame([
            {
                "contact_email": "hr@corp-a.com",
                "application_url": "https://corp-a.com/apply",
                "company_website": "https://corp-a.com",
            },
            {
                "contact_email": "scam@gmail.com",
                "application_url": "https://unknown-site.xyz/job",
                "company_website": None,
            },
        ])

        with patch("src.legitimacy.CACHE_DB_PATH", tmp_path / "test_cache.sqlite"), \
             patch("src.legitimacy.CACHE_DIR", tmp_path):
            result_df = batch_check(df)
            assert len(result_df) == 2
            assert "contact_domain" in result_df.columns
            assert "free_email" in result_df.columns
            assert result_df.loc[0, "contact_domain"] == "corp-a.com"
            assert result_df.loc[1, "contact_domain"] == "gmail.com"
            assert bool(result_df.loc[1, "free_email"]) is True



# ===================================================================
# Live network test
# ===================================================================
@pytest.mark.live
def test_live_legitimacy_check():
    """Live test performing actual network calls against a known stable domain (python.org)."""
    result = check_posting(
        contact_email="help@python.org",
        application_url="https://www.python.org/jobs",
        company_website="https://www.python.org",
    )

    assert result["contact_domain"] == "python.org"
    assert result["company_domain"] == "python.org"
    assert result["domain_match"] is True
    assert result["free_email"] is False
    assert result["mx_present"] is True
    assert result["https_valid"] is True
    assert result["domain_age_days"] is not None and result["domain_age_days"] > 5000
    assert result["cert_days_left"] is not None and result["cert_days_left"] >= 0
