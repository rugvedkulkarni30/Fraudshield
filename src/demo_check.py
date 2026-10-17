"""
demo_check.py — CLI tool to check job posting legitimacy.
=========================================================
Usage:
    python src/demo_check.py --email hr@company.com --url https://jobs.lever.co/company --website https://company.com
    or positional:
    python src/demo_check.py hr@company.com https://jobs.lever.co/company https://company.com
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.legitimacy import check_posting


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check job posting legitimacy indicators (domain, age, ATS, MX, HTTPS)."
    )
    parser.add_argument(
        "positional_args",
        nargs="*",
        help="Optional positional args: [contact_email] [application_url] [company_website]",
    )
    parser.add_argument("--email", "-e", default=None, help="Contact email address")
    parser.add_argument("--url", "-u", default=None, help="Application or job post URL")
    parser.add_argument("--website", "-w", default=None, help="Company official website URL")

    args = parser.parse_args()

    email = args.email
    url = args.url
    website = args.website

    if args.positional_args:
        if len(args.positional_args) >= 1 and not email:
            email = args.positional_args[0]
        if len(args.positional_args) >= 2 and not url:
            url = args.positional_args[1]
        if len(args.positional_args) >= 3 and not website:
            website = args.positional_args[2]

    print("Running legitimacy checks...")
    result = check_posting(contact_email=email, application_url=url, company_website=website)

    print("\n" + "=" * 55)
    print(f"{'LEGITIMACY CHECK RESULT':^55}")
    print("=" * 55)
    print(f"{'Indicator':<25} | {'Value':<27}")
    print("-" * 55)

    display_labels = [
        ("contact_domain", "Contact Domain"),
        ("company_domain", "Company Domain"),
        ("domain_age_days", "Domain Age (Days)"),
        ("domain_match", "Domain Match"),
        ("free_email", "Free Webmail"),
        ("is_ats_domain", "ATS Domain (Allowlist)"),
        ("mx_present", "MX Records Present"),
        ("https_valid", "HTTPS Valid"),
        ("cert_days_left", "Cert Days Left"),
    ]

    for key, label in display_labels:
        val = result.get(key)
        val_str = "None (Unknown)" if val is None else str(val)
        print(f"{label:<25} | {val_str:<27}")

    print("-" * 55)
    errors = result.get("lookup_errors", [])
    if errors:
        print("Lookup Warnings / Errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("Lookup Errors: None")
    print("=" * 55)


if __name__ == "__main__":
    main()
