# scrape_data.py
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

BASE = "https://web-scraping.dev"
PRODUCTS_URL = f"{BASE}/products"
REVIEWS_URL = f"{BASE}/reviews"
TESTIMONIALS_URL = f"{BASE}/testimonials"

TESTIMONIALS_API = f"{BASE}/api/testimonials"
TESTIMONIALS_TOKEN = "secret123"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


@dataclass
class Product:
    name: str
    price: Optional[float]
    product_url: str
    page_url: str


@dataclass
class Review:
    title: str
    text: str
    rating: Optional[float]
    date_raw: str
    date_iso: Optional[str]
    page: int
    source_url: str


@dataclass
class Testimonial:
    text: str
    rating: Optional[int]
    page: int
    source_url: str


def fetch(url: str, params: Optional[Dict[str, Any]] = None, headers_extra: Optional[Dict[str, str]] = None) -> requests.Response:
    headers = dict(HEADERS)
    if headers_extra:
        headers.update(headers_extra)
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r


def parse_date_to_iso(date_str: str) -> Optional[str]:
    try:
        dt = dateparser.parse(date_str, fuzzy=True)
        return dt.date().isoformat() if dt else None
    except Exception:
        return None


def scrape_products() -> List[Product]:
    products = []
    page = 1

    while True:
        page_url = f"{PRODUCTS_URL}?page={page}"
        soup = BeautifulSoup(fetch(page_url).text, "lxml")

        links = soup.select("h3 a")
        if not links:
            break

        for a in links:
            name = a.get_text(strip=True)
            product_url = urljoin(BASE, a.get("href", ""))
            text_blob = a.find_parent().get_text(" ", strip=True)
            price = None

            for token in text_blob.replace("$", " ").split():
                try:
                    if "." in token:
                        price = float(token)
                        break
                except Exception:
                    pass

            products.append(
                Product(
                    name=name,
                    price=price,
                    product_url=product_url,
                    page_url=page_url,
                )
            )

        page += 1
        time.sleep(0.2)

    return products


def scrape_testimonials() -> List[Testimonial]:
    testimonials = []
    page = 1

    while True:
        try:
            html = fetch(
                TESTIMONIALS_API,
                params={"page": page},
                headers_extra={
                    "Referer": TESTIMONIALS_URL,
                    "X-Secret-Token": TESTIMONIALS_TOKEN,
                },
            ).text
        except Exception:
            break

        soup = BeautifulSoup(html, "lxml")
        cards = soup.select(".testimonial")
        if not cards:
            break

        for card in cards:
            text_el = card.select_one(".text")
            text = text_el.get_text(" ", strip=True) if text_el else card.get_text(" ", strip=True)
            stars = card.select(".rating svg")
            rating = len(stars) if stars else None

            testimonials.append(
                Testimonial(
                    text=text,
                    rating=rating,
                    page=page,
                    source_url=f"{TESTIMONIALS_API}?page={page}",
                )
            )

        page += 1
        time.sleep(0.2)

    return testimonials


def scrape_reviews() -> List[Review]:
    """
    Reviews are loaded via GraphQL at /api/graphql.
    We query the `reviews` field and paginate until no more results.
    """
    reviews: List[Review] = []
    page = 1

    gql_url = f"{BASE}/api/graphql"

    # We'll request a page size per call (safe and fast)
    first = 50
    after = None

    query = """
    query ReviewsPage($first: Int!, $after: String) {
      reviews(first: $first, after: $after) {
        pageInfo { hasNextPage endCursor }
        edges {
          node {
            id
            date
            rating
            text
          }
        }
      }
    }
    """

    while True:
        variables = {"first": first, "after": after}
        resp = requests.post(
            gql_url,
            json={"query": query, "variables": variables},
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Safety: stop if server returns errors
        if "errors" in data and data["errors"]:
            print("GraphQL errors:", data["errors"])
            break

        block = (data.get("data") or {}).get("reviews") or {}
        edges = block.get("edges") or []
        if not edges:
            break

        for e in edges:
            node = e.get("node") or {}
            date_raw = str(node.get("date") or "")
            date_iso = parse_date_to_iso(date_raw) if date_raw else None

            rating_val = node.get("rating")
            rating = None
            try:
                if rating_val is not None:
                    rating = float(rating_val)
            except Exception:
                rating = None

            text = str(node.get("text") or "")
            rid = str(node.get("id") or "")

            reviews.append(
                Review(
                    title="",
                    text=text,
                    rating=rating,
                    date_raw=date_raw,
                    date_iso=date_iso,
                    page=page,
                    source_url=f"{gql_url} (after={after}) id={rid}",
                )
            )

        page_info = block.get("pageInfo") or {}
        has_next = bool(page_info.get("hasNextPage"))
        after = page_info.get("endCursor")

        if not has_next or not after:
            break

        page += 1
        time.sleep(0.2)

    return reviews




def main():
    os.makedirs("data", exist_ok=True)

    print("Scraping products...")
    products = scrape_products()
    print("Scraping testimonials...")
    testimonials = scrape_testimonials()
    print("Scraping reviews...")
    reviews = scrape_reviews()

    output = {
        "products": [asdict(p) for p in products],
        "testimonials": [asdict(t) for t in testimonials],
        "reviews": [asdict(r) for r in reviews],
        "meta": {
            "products_url": PRODUCTS_URL,
            "testimonials_url": TESTIMONIALS_URL,
            "reviews_url": REVIEWS_URL,
        },
    }

    with open("data/scraped_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("✅ Saved data/scraped_data.json")
    print(f"Counts: products={len(products)}, testimonials={len(testimonials)}, reviews={len(reviews)}")


if __name__ == "__main__":
    main()
