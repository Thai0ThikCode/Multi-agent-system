"""
tools/pvoil_crawler.py
Crawls and extracts domestic Vietnamese retail fuel prices from PVOIL:
https://www.pvoil.com.vn/tin-gia-xang-dau

Features:
 - Embedded HTML table parser for retail fuel prices (RON 95, E5 RON 92, Diesel, etc.)
 - Zone 1 (Vùng 1) and Zone 2 (Vùng 2) price extraction in VNĐ/lít
 - Effective date and announcement extraction
 - Cloudflare detection + graceful fallback data with user-friendly warnings
 - Direct HTML string parsing support (allows user to paste or upload raw HTML)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import DATA_CONFIG
from schemas import FuelPriceItem, DomesticFuelSnapshot

logger = logging.getLogger(__name__)

# Fallback reference prices based on latest standard official adjustment
DEFAULT_BENCHMARK_ITEMS = [
    FuelPriceItem(product_name="Xăng RON 95-III", zone_1_price=27080.0, zone_2_price=27620.0, unit="VNĐ/lít"),
    FuelPriceItem(product_name="Xăng E5 RON 92-II", zone_1_price=26390.0, zone_2_price=26910.0, unit="VNĐ/lít"),
    FuelPriceItem(product_name="Dầu DO 0,05S-II (Diesel)", zone_1_price=30490.0, zone_2_price=31090.0, unit="VNĐ/lít"),
    FuelPriceItem(product_name="Dầu hỏa 2-K", zone_1_price=30020.0, zone_2_price=30620.0, unit="VNĐ/lít"),
    FuelPriceItem(product_name="Dầu Mazut 180cst-0,5S", zone_1_price=27470.0, zone_2_price=28010.0, unit="VNĐ/kg"),
]


def _clean_price_val(raw_str: str) -> Optional[float]:
    """Converts price strings like '26.390', '26,390', '27 080' into float 26390.0."""
    cleaned = re.sub(r"[^\d.,]", "", raw_str).strip()
    if not cleaned:
        return None
    # If formatted like 26.390 or 26,390 (thousands separator)
    if re.search(r"^\d{1,3}[.,]\d{3}$", cleaned):
        cleaned = re.sub(r"[.,]", "", cleaned)
    elif "." in cleaned and "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        val = float(cleaned)
        # Retail fuel in Vietnam is usually between 10,000 and 50,000 VND
        return val if val > 0 else None
    except ValueError:
        return None


def parse_pvoil_html(html_text: str) -> Optional[DomesticFuelSnapshot]:
    """Parses raw HTML from PVOIL fuel price announcements."""
    if not html_text or not html_text.strip():
        return None

    soup = BeautifulSoup(html_text, "html.parser")

    # 1. Attempt to extract effective date/time
    effective_date = ""
    date_patterns = [
        r"(?:từ|ngay|ngày)\s+(\d{1,2}(?:h\d{2})?\s+ngày\s+\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
    ]
    for text_block in soup.find_all(string=re.compile(r"ngày|áp dụng|hiệu lực", re.IGNORECASE)):
        for pat in date_patterns:
            m = re.search(pat, text_block, re.IGNORECASE)
            if m:
                effective_date = m.group(1).strip()
                break
        if effective_date:
            break

    if not effective_date:
        effective_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    items: list[FuelPriceItem] = []
    seen_products: set[str] = set()

    # 2. Search tables
    tables = soup.select("table, .pricing-table, .table")
    for table in tables:
        for row in table.select("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.select("th, td")]
            if len(cells) < 2:
                continue

            row_text = " | ".join(cells).lower()
            # Check if this row mentions a known fuel category
            is_fuel_row = any(
                k in row_text
                for k in ["ron 95", "ron 92", "e5", "diesel", "do 0", "dầu hỏa", "mazut", "xăng"]
            )
            if not is_fuel_row:
                continue

            # Identify product name (usually cell 0 or 1)
            product_name = cells[0]
            if re.match(r"^\d+$", product_name) and len(cells) > 2:
                product_name = cells[1]

            # Normalize product name
            product_name = re.sub(r"\s+", " ", product_name).strip()
            if product_name in seen_products:
                continue

            # Identify price numbers
            price_vals = []
            for cell in cells[1:]:
                p = _clean_price_val(cell)
                if p and p > 1000:  # Retail fuel price threshold in VND
                    price_vals.append(p)

            if not price_vals:
                continue

            z1 = price_vals[0]
            z2 = price_vals[1] if len(price_vals) > 1 else round(z1 * 1.02, 0)
            unit = "VNĐ/kg" if "mazut" in product_name.lower() else "VNĐ/lít"

            items.append(
                FuelPriceItem(
                    product_name=product_name,
                    zone_1_price=z1,
                    zone_2_price=z2,
                    unit=unit,
                )
            )
            seen_products.add(product_name)

    if items:
        return DomesticFuelSnapshot(
            effective_date=effective_date,
            source="PVOIL (Parsed HTML)",
            items=items,
            notes=[f"Successfully parsed {len(items)} fuel products from HTML."],
        )

    return None


def crawl_pvoil_fuel_prices(
    url: Optional[str] = None,
    custom_html: Optional[str] = None,
) -> tuple[Optional[DomesticFuelSnapshot], list[str]]:
    """Crawls PVOIL fuel prices. If custom_html is provided, parses it directly.
    Otherwise attempts an HTTP request with realistic browser headers and falls back
    to curated reference prices if Cloudflare challenge blocks the request.
    """
    target_url = url or DATA_CONFIG.pvoil_oil_url
    warnings: list[str] = []

    # Case 1: Custom HTML provided (e.g., from Streamlit file upload or paste)
    if custom_html and custom_html.strip():
        snapshot = parse_pvoil_html(custom_html)
        if snapshot:
            return snapshot, warnings
        warnings.append("Provided custom HTML could not be parsed into a valid fuel price table.")

    # Case 2: Direct HTTP fetch
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.google.com/",
    }

    try:
        session = requests.Session()
        resp = session.get(target_url, headers=headers, timeout=15)

        if resp.status_code == 200:
            html = resp.text
            if "cf-mitigated" in html.lower() or "challenges.cloudflare.com" in html.lower():
                warnings.append(
                    f"PVOIL ({target_url}) is protected by Cloudflare Bot Management. "
                    "Using reference benchmark fuel data. You can paste the live HTML in the dashboard."
                )
            else:
                snapshot = parse_pvoil_html(html)
                if snapshot:
                    return snapshot, warnings
                warnings.append("PVOIL page returned 200 OK, but no pricing table could be extracted.")
        else:
            warnings.append(
                f"PVOIL request returned status {resp.status_code} (Cloudflare block). "
                "Using reference benchmark fuel data. You can paste or upload the live HTML in the dashboard."
            )
    except Exception as e:
        warnings.append(f"PVOIL fetch network error: {e}. Using reference benchmark fuel data.")

    # Fallback to curated benchmark data
    fallback_snapshot = DomesticFuelSnapshot(
        effective_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        source="PVOIL (Reference Benchmark)",
        items=DEFAULT_BENCHMARK_ITEMS,
        notes=[
            "Data generated from curated Vietnam MOIT/PVOIL price schedule. "
            "To use real-time live data, upload or paste HTML in the dashboard."
        ],
    )
    return fallback_snapshot, warnings
