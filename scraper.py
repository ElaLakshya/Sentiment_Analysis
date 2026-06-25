"""
Task 1: fetch the last `target_count` tweets from a profile URL.

Key design decisions:
- Runs HEADED (visible browser) by default. X serves different content in
  headless mode even with a stealth browser - specifically it shows "For You"
  recommendations instead of the actual profile feed. Headed mode gets the
  real profile page every time.
- Explicitly navigates to the profile URL + waits for actual tweet articles
  to appear before scrolling, rather than sleeping a fixed amount of time.
- Anti-hallucination filter: only saves tweets whose URL matches the target
  handle, which filters out retweets and any suggested/explore content.
"""

import re
import time
from datetime import datetime, timezone

from cloakbrowser import launch_persistent_context

from config import PROFILE_DIR, DEFAULT_TWEET_TARGET
from db import init_db, insert_tweet

TWEET_SELECTOR    = 'article[data-testid="tweet"]'
MAX_STAGNANT_SCROLLS = 8


def _parse_tweet_id(href: str):
    m = re.search(r"/status/(\d+)", href or "")
    return m.group(1) if m else None


def _extract_tweet(article, profile_handle: str):
    try:
        text_el = article.query_selector('div[data-testid="tweetText"]')
        text    = text_el.inner_text() if text_el else ""

        time_el    = article.query_selector("time")
        created_at = time_el.get_attribute("datetime") if time_el else None

        link_el = article.query_selector('a[href*="/status/"]')
        href    = link_el.get_attribute("href") if link_el else None

        # Only keep tweets that actually belong to this profile handle.
        # This filters retweets, suggested accounts, and explore content.
        if not href or f"/{profile_handle.lower()}/status/" not in href.lower():
            return None

        tweet_id = _parse_tweet_id(href)
        if not tweet_id:
            return None

        return {
            "tweet_id":       tweet_id,
            "profile_handle": profile_handle,
            "text":           text,
            "created_at":     created_at,
            "tweet_url":      f"https://x.com{href}",
            "scraped_at":     datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


def _wait_for_tweets(page, timeout_sec: int = 20) -> bool:
    """Wait until at least one tweet article appears on the page."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        articles = page.query_selector_all(TWEET_SELECTOR)
        if articles:
            return True
        time.sleep(1)
    return False


def scrape_profile(
    profile_url:  str,
    target_count: int  = DEFAULT_TWEET_TARGET,
    profile_dir:  str  = PROFILE_DIR,
    headless:     bool = False,   # ← headed by default: X serves real feeds
) -> int:
    profile_handle = profile_url.rstrip("/").split("/")[-1]
    init_db()

    context = launch_persistent_context(profile_dir, headless=headless)
    page    = context.pages[0] if context.pages else context.new_page()

    # Navigate to the profile page
    print(f"  → Navigating to {profile_url} ...")
    page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
    time.sleep(2)

    # X sometimes lands on a "For You" tab or shows a login prompt even with
    # cookies. Scroll to top and wait for actual tweet articles.
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(1)

    tweets_found = _wait_for_tweets(page, timeout_sec=20)
    if not tweets_found:
        print("  ⚠ No tweet articles appeared after 20s. The session may have expired.")
        print("    Re-run: python import_cookies.py cookie_export.json")
        context.close()
        return 0

    seen_ids  = set()
    collected = 0
    stagnant  = 0

    while collected < target_count and stagnant < MAX_STAGNANT_SCROLLS:
        articles     = page.query_selector_all(TWEET_SELECTOR)
        new_this_pass = 0

        for article in articles:
            data = _extract_tweet(article, profile_handle)
            if data and data["tweet_id"] not in seen_ids:
                seen_ids.add(data["tweet_id"])
                insert_tweet(data)
                collected    += 1
                new_this_pass += 1
                if collected >= target_count:
                    break

        stagnant = 0 if new_this_pass else stagnant + 1
        if stagnant > 0:
            print(f"  → No new tweets this scroll ({stagnant}/{MAX_STAGNANT_SCROLLS}), scrolling deeper...")

        page.mouse.wheel(0, 2800)
        time.sleep(2)   # slightly longer pause so X's feed has time to load

    context.close()
    return collected


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <profile_url> [count]")
        sys.exit(1)
    url   = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TWEET_TARGET
    n     = scrape_profile(url, target_count=count)
    print(f"Scraped {n} tweets from {url}")