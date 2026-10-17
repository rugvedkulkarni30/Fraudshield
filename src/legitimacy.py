"""
legitimacy.py — External legitimacy verification for job postings.
==================================================================
Performs privacy-preserving domain-level legitimacy checks on:
- contact_email (domain only, discarding local-part)
- application_url
- company_website

Features:
- RDAP lookup with python-whois fallback for domain age.
- SQLite caching at data/cache/domains.sqlite (30-day expiry).
- 5s timeout with 2 retries on network operations.
- Rule: Unknown values are returned as None, never 0 or False.
- Privacy: Never stores or returns registrant PII or email local-parts.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import socket
import ssl
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import dns.exception
import dns.resolver
import pandas as pd
import requests
import tldextract
import whois

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DB_PATH = CACHE_DIR / "domains.sqlite"
CACHE_EXPIRY_SECONDS = 30 * 86400  # 30 days
NETWORK_TIMEOUT = 5  # seconds
MAX_RETRIES = 2

# Curated lists
FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "ymail.com",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "msn.com",
    "aol.com",
    "aim.com",
    "icloud.com",
    "me.com",
    "mac.com",
    "mail.com",
    "zoho.com",
    "protonmail.com",
    "proton.me",
    "yandex.com",
    "yandex.ru",
    "gmx.com",
    "gmx.net",
    "mail.ru",
    "fastmail.com",
    "inbox.com",
    "qq.com",
    "163.com",
    "126.com",
    "rediffmail.com",
}

ATS_DOMAINS = {
    "greenhouse.io",
    "lever.co",
    "workday.com",
    "myworkdayjobs.com",
    "smartrecruiters.com",
    "icims.com",
    "taleo.net",
    "bamboohr.com",
    "ashbyhq.com",
    "jobvite.com",
    "recruitee.com",
    "workable.com",
    "breezy.hr",
    "jazzhr.com",
    "applytojob.com",
    "pinpointhq.com",
    "rippling.com",
    "gusto.com",
    "paylocity.com",
    "ultipro.com",
    "adp.com",
    "cornerstoneondemand.com",
    "brassring.com",
    "successfactors.com",
    "teamtailor.com",
    "freshteam.com",
    "personio.com",
    "join.com",
    "indeed.com",
    "linkedin.com",
    "cv-library.co.uk",
}

_EXTRACTOR = tldextract.TLDExtract(cache_dir=str(CACHE_DIR / "tldextract_cache"))


def _init_cache_db() -> None:
    """Initialize SQLite cache database and table."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS domain_cache (
                domain TEXT PRIMARY KEY,
                mx_present INTEGER,
                creation_date TEXT,
                https_valid INTEGER,
                cert_days_left INTEGER,
                cert_not_after TEXT,
                cached_at REAL
            )
            """
        )
        conn.commit()


def _get_cached_domain(domain: str) -> dict[str, Any] | None:
    """Retrieve non-expired cached domain attributes."""
    if not CACHE_DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT mx_present, creation_date, https_valid, cert_days_left, cert_not_after, cached_at
                FROM domain_cache WHERE domain = ?
                """,
                (domain.lower(),),
            )
            row = cur.fetchone()
            if not row:
                return None
            mx_present, creation_date, https_valid, cert_days_left, cert_not_after, cached_at = row
            if (time.time() - cached_at) > CACHE_EXPIRY_SECONDS:
                return None

            # Calculate fresh cert_days_left if cert_not_after is stored
            if cert_not_after:
                try:
                    not_after_dt = datetime.datetime.fromisoformat(cert_not_after)
                    now_dt = datetime.datetime.now(datetime.timezone.utc)
                    if not_after_dt.tzinfo is None:
                        not_after_dt = not_after_dt.replace(tzinfo=datetime.timezone.utc)
                    cert_days_left = max(0, (not_after_dt - now_dt).days)
                except Exception:
                    pass

            return {
                "mx_present": bool(mx_present) if mx_present is not None else None,
                "creation_date": creation_date,
                "https_valid": bool(https_valid) if https_valid is not None else None,
                "cert_days_left": cert_days_left,
                "cert_not_after": cert_not_after,
            }
    except Exception:
        return None


def _set_cached_domain(domain: str, data: dict[str, Any]) -> None:
    """Insert or update domain cache entry."""
    try:
        _init_cache_db()
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO domain_cache
                (domain, mx_present, creation_date, https_valid, cert_days_left, cert_not_after, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    domain.lower(),
                    int(data["mx_present"]) if data.get("mx_present") is not None else None,
                    data.get("creation_date"),
                    int(data["https_valid"]) if data.get("https_valid") is not None else None,
                    data.get("cert_days_left"),
                    data.get("cert_not_after"),
                    time.time(),
                ),
            )
            conn.commit()
    except Exception:
        pass


def extract_registered_domain(text: str | None) -> str | None:
    """
    Extract the registered domain (e.g., example.com) from an email, URL, or domain string.
    Discards any email local part and registrant info.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned or cleaned.lower() in ("nan", "none", "null"):
        return None

    # Discard email local-part if present
    if "@" in cleaned:
        cleaned = cleaned.split("@")[-1].strip()

    # If it's a URL or contains protocol/slashes
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        cleaned = parsed.netloc or parsed.path
    elif "/" in cleaned:
        cleaned = cleaned.split("/")[0].strip()

    # Remove port if present
    if ":" in cleaned:
        cleaned = cleaned.split(":")[0].strip()

    ext = _EXTRACTOR(cleaned)
    domain = getattr(ext, "top_domain_under_public_suffix", None) or getattr(ext, "registered_domain", "")
    return domain.lower() if domain else None



def check_mx_records(domain: str, errors: list[str]) -> bool | None:
    """Check for MX DNS records with timeout and retries."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resolver = dns.resolver.Resolver()
            resolver.lifetime = NETWORK_TIMEOUT
            resolver.timeout = NETWORK_TIMEOUT
            answers = resolver.resolve(domain, "MX")
            return len(answers) > 0
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            return False
        except dns.exception.Timeout:
            if attempt == MAX_RETRIES:
                errors.append(f"DNS MX timeout for {domain}")
                return None
            time.sleep(0.5)
        except Exception as e:
            if attempt == MAX_RETRIES:
                errors.append(f"DNS MX lookup error for {domain}: {e}")
                return None
            time.sleep(0.5)
    return None


def fetch_domain_creation_date(domain: str, errors: list[str]) -> str | None:
    """
    Fetch domain creation date using RDAP with python-whois fallback.
    Returns ISO date string (YYYY-MM-DD) or None.
    Discards all registrant personal details.
    """
    # 1. Try RDAP
    for attempt in range(MAX_RETRIES + 1):
        try:
            url = f"https://rdap.org/domain/{domain}"
            resp = requests.get(
                url,
                timeout=NETWORK_TIMEOUT,
                headers={"Accept": "application/rdap+json, application/json"},
            )
            if resp.status_code == 200:
                rdap_data = resp.json()
                for event in rdap_data.get("events", []):
                    action = event.get("eventAction", "").lower()
                    if action in ("registration", "created"):
                        event_date = event.get("eventDate")
                        if event_date:
                            # Normalize date string
                            dt = datetime.datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                            return dt.date().isoformat()
            elif resp.status_code == 404:
                # Not found on RDAP, try whois fallback
                break
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES:
                errors.append(f"RDAP timeout for {domain}")
        except Exception as e:
            if attempt == MAX_RETRIES:
                errors.append(f"RDAP error for {domain}: {e}")

    # 2. Fallback to python-whois
    for attempt in range(MAX_RETRIES + 1):
        try:
            w = whois.whois(domain)
            creation_date = getattr(w, "creation_date", None)
            if creation_date:
                if isinstance(creation_date, (list, tuple)):
                    creation_date = min(
                        [d for d in creation_date if isinstance(d, (datetime.datetime, datetime.date))]
                    )
                if isinstance(creation_date, datetime.datetime):
                    return creation_date.date().isoformat()
                if isinstance(creation_date, datetime.date):
                    return creation_date.isoformat()
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                errors.append(f"WHOIS fallback error for {domain}: {e}")
            time.sleep(0.5)

    return None


def check_https_and_cert(domain: str, errors: list[str]) -> tuple[bool | None, int | None, str | None]:
    """
    Validate HTTPS certificate and return (https_valid, cert_days_left, cert_not_after_iso).
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=NETWORK_TIMEOUT) as sock:
                with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    not_after_str = cert.get("notAfter")
                    if not_after_str:
                        # Format example: 'May 15 12:00:00 2026 GMT'
                        not_after_dt = datetime.datetime.strptime(
                            not_after_str, "%b %d %H:%M:%S %Y %Z"
                        ).replace(tzinfo=datetime.timezone.utc)
                        now_dt = datetime.datetime.now(datetime.timezone.utc)
                        days_left = max(0, (not_after_dt - now_dt).days)
                        return True, days_left, not_after_dt.isoformat()
                    return True, None, None
        except ssl.SSLCertVerificationError as e:
            errors.append(f"HTTPS SSL verification error for {domain}: {e}")
            return False, None, None
        except (socket.timeout, TimeoutError):
            if attempt == MAX_RETRIES:
                errors.append(f"HTTPS connection timeout for {domain}")
                return None, None, None
            time.sleep(0.5)
        except (ConnectionRefusedError, socket.gaierror, OSError) as e:
            if attempt == MAX_RETRIES:
                errors.append(f"HTTPS connection error for {domain}: {e}")
                return False, None, None
            time.sleep(0.5)
        except Exception as e:
            if attempt == MAX_RETRIES:
                errors.append(f"HTTPS error for {domain}: {e}")
                return None, None, None
            time.sleep(0.5)

    return None, None, None


def check_posting(
    contact_email: str | None = None,
    application_url: str | None = None,
    company_website: str | None = None,
) -> dict[str, Any]:
    """
    Perform legitimacy analysis on a single posting.

    Returns dict with keys:
    - contact_domain: str | None
    - company_domain: str | None
    - domain_age_days: int | None
    - domain_match: bool | None
    - free_email: bool | None
    - is_ats_domain: bool | None
    - mx_present: bool | None
    - https_valid: bool | None
    - cert_days_left: int | None
    - lookup_errors: list[str]
    """
    errors: list[str] = []

    # 1. Extract domains
    contact_domain = extract_registered_domain(contact_email)

    company_domain_from_web = extract_registered_domain(company_website)
    company_domain_from_app = extract_registered_domain(application_url)

    # ATS detection: check if application_url or company_website uses an ATS domain
    is_ats_domain: bool | None = None
    has_urls = bool(application_url or company_website)
    if has_urls:
        ats_candidates = {d for d in (company_domain_from_web, company_domain_from_app) if d}
        is_ats_domain = any(cand in ATS_DOMAINS for cand in ats_candidates)

    # company_domain preference: company_website first, then application_url (if not ATS, or fallback)
    company_domain: str | None = None
    if company_domain_from_web:
        company_domain = company_domain_from_web
    elif company_domain_from_app and company_domain_from_app not in ATS_DOMAINS:
        company_domain = company_domain_from_app
    elif company_domain_from_app:
        company_domain = company_domain_from_app

    # 2. Free email
    free_email: bool | None = None
    if contact_domain is not None:
        free_email = contact_domain in FREE_EMAIL_DOMAINS

    # 3. Domain match
    domain_match: bool | None = None
    if contact_domain is not None and company_domain is not None:
        domain_match = (contact_domain == company_domain)

    # 4. Target domain for age & https lookups
    # Prioritize company_domain (if not ATS), else contact_domain (if not free), else available
    target_domain: str | None = None
    if company_domain and company_domain not in ATS_DOMAINS:
        target_domain = company_domain
    elif contact_domain and not free_email:
        target_domain = contact_domain
    elif company_domain:
        target_domain = company_domain
    elif contact_domain:
        target_domain = contact_domain

    # Domain-level lookups
    mx_present: bool | None = None
    domain_age_days: int | None = None
    https_valid: bool | None = None
    cert_days_left: int | None = None

    # Check MX for contact_domain
    if contact_domain:
        cached = _get_cached_domain(contact_domain)
        if cached and cached.get("mx_present") is not None:
            mx_present = cached["mx_present"]
        else:
            mx_present = check_mx_records(contact_domain, errors)
            if mx_present is not None:
                _set_cached_domain(contact_domain, {"mx_present": mx_present})

    # Check age and HTTPS for target_domain
    if target_domain:
        cached = _get_cached_domain(target_domain)
        creation_date_str = cached.get("creation_date") if cached else None
        cached_https = cached.get("https_valid") if cached else None
        cached_cert_days = cached.get("cert_days_left") if cached else None
        cert_not_after = cached.get("cert_not_after") if cached else None

        # Domain age
        if not creation_date_str:
            creation_date_str = fetch_domain_creation_date(target_domain, errors)

        if creation_date_str:
            try:
                c_date = datetime.date.fromisoformat(creation_date_str)
                domain_age_days = max(0, (datetime.date.today() - c_date).days)
            except Exception:
                domain_age_days = None

        # HTTPS / SSL Cert
        if cached_https is not None:
            https_valid = cached_https
            cert_days_left = cached_cert_days
        else:
            https_valid, cert_days_left, cert_not_after = check_https_and_cert(target_domain, errors)

        # Update cache
        cache_update: dict[str, Any] = {}
        if creation_date_str is not None:
            cache_update["creation_date"] = creation_date_str
        if https_valid is not None:
            cache_update["https_valid"] = https_valid
            cache_update["cert_days_left"] = cert_days_left
            cache_update["cert_not_after"] = cert_not_after
        if target_domain == contact_domain and mx_present is not None:
            cache_update["mx_present"] = mx_present

        if cache_update:
            _set_cached_domain(target_domain, cache_update)

    return {
        "contact_domain": contact_domain,
        "company_domain": company_domain,
        "domain_age_days": domain_age_days,
        "domain_match": domain_match,
        "free_email": free_email,
        "is_ats_domain": is_ats_domain,
        "mx_present": mx_present,
        "https_valid": https_valid,
        "cert_days_left": cert_days_left,
        "lookup_errors": errors,
    }


def batch_check(
    df: pd.DataFrame,
    columns: Mapping[str, str] | Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Run legitimacy check over a pandas DataFrame in batch.

    Parameters:
    - df: pandas DataFrame containing posting records.
    - columns: dict mapping {'email': col, 'url': col, 'website': col} or
               a 3-element sequence (email_col, url_col, website_col).
    """
    email_col, url_col, website_col = None, None, None

    if isinstance(columns, Mapping):
        email_col = columns.get("email") or columns.get("contact_email")
        url_col = columns.get("url") or columns.get("application_url")
        website_col = columns.get("website") or columns.get("company_website")
    elif isinstance(columns, Sequence) and len(columns) >= 3:
        email_col, url_col, website_col = columns[0], columns[1], columns[2]
    else:
        # Default heuristics for column names in EMSCAD or generic datasets
        for col in df.columns:
            lcol = col.lower()
            if not email_col and ("email" in lcol or "contact" in lcol):
                email_col = col
            elif not url_col and ("url" in lcol or "application" in lcol or "apply" in lcol):
                url_col = col
            elif not website_col and ("website" in lcol or "company_profile" in lcol or "site" in lcol):
                website_col = col

    results = []
    for _, row in df.iterrows():
        email = str(row[email_col]) if email_col and email_col in df.columns and pd.notna(row[email_col]) else None
        url = str(row[url_col]) if url_col and url_col in df.columns and pd.notna(row[url_col]) else None
        website = str(row[website_col]) if website_col and website_col in df.columns and pd.notna(row[website_col]) else None

        res = check_posting(contact_email=email, application_url=url, company_website=website)
        results.append(res)

    return pd.DataFrame(results)
