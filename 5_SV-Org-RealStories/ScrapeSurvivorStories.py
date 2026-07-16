"""Scrape survivor stories from thesurvivorstrust.org using Playwright."""

import time
import pandas as pd
from bs4 import BeautifulSoup
from StoryURLS import STORY_URLS
from playwright.sync_api import sync_playwright

def parse_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.select_one("h1.elementor-heading-title")
    title = title.get_text(strip=True) if title else None

    date = soup.select_one("li[itemprop='datePublished'] time")
    date = date.get_text(strip=True) if date else None

    content = soup.select_one(
        "div.elementor-widget-theme-post-content div.elementor-widget-container"
    )
    # fallback: any post content container
    if not content:
        content = soup.select_one("div.entry-content") or soup.select_one("article")

    paragraphs = []
    if content:
        paragraphs = [
            p.get_text(strip=True)
            for p in content.find_all("p")
            if p.get_text(strip=True)
        ]

    author = None
    if paragraphs and paragraphs[0].lower().startswith("by "):
        author = paragraphs[0].replace("By ", "").strip()
        paragraphs = paragraphs[1:]

    return {
        "story_url": url,
        "title": title,
        "date": date,
        "author": author,
        "blurb": "\n\n".join(paragraphs),
        "num_paragraphs": len(paragraphs),
    }


def main():
    urls = STORY_URLS.split()
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for idx, url in enumerate(urls):
            print(f"[{idx+1}/{len(urls)}] {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(1)  # let any lazy-loaded content settle
                row = parse_page(page.content(), url)
                if not row["blurb"]:
                    print(f"  WARNING: no text extracted")
                rows.append(row)
            except Exception as e:
                print(f"  ERROR: {e}")
                rows.append({"story_url": url, "title": None, "date": None,
                              "author": None, "blurb": None, "num_paragraphs": 0})

        browser.close()

    df = pd.DataFrame(rows)
    df.to_csv("Data/SurvivorsStories.csv", index=False)
    print(f"\nDone. {len(df)} rows. Saved to SurvivorsStories.csv")
    print(df[["title", "num_paragraphs"]].to_string())

if __name__ == "__main__":
    main()