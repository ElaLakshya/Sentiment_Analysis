import sys
import time
import argparse
from collections import Counter

from config import PROFILE_DIR, DEFAULT_TWEET_TARGET, BITNET_MODEL_NAME
import db
from scraper import scrape_profile
from bitnet_client import analyze_tweet
from fasttext_client import classify_taxonomy

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("Error: 'rich' is not installed. Run: pip install rich")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Run: pip install matplotlib")
    sys.exit(1)


def show_pie_chart(category_counts: dict, profile_name: str):
    """Pie chart with all labels in a legend below — no text on the slices."""
    if not category_counts:
        print("\nNo taxonomy categories to plot.")
        return

    sorted_data   = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*sorted_data)
    total = sum(sizes)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 9))

    # Create number labels for the pie chart slices (1, 2, 3...)
    slice_numbers = [str(i) for i in range(1, len(labels) + 1)]

    wedges, texts = ax.pie(
        sizes,
        labels=slice_numbers,
        labeldistance=1.05,  # Places the numbers just outside the pie edge
        startangle=140,
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
        textprops={"color": "white", "weight": "bold", "fontsize": 10},
    )

    # Legend: coloured square | Number. category name | percentage
    legend_labels = [
        f"{i}. {lbl}  ({sz/total*100:.1f}%)"
        for i, (lbl, sz) in enumerate(zip(labels, sizes), start=1)
    ]
    ax.legend(
        wedges, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        fontsize=9,
        framealpha=0.15,
        labelcolor="white",
    )

    ax.set_title(
        f"Topic Analysis — @{profile_name}\n({total} tweets · Google Taxonomy)",
        fontsize=13, weight="bold", pad=16, color="white",
    )
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    plt.tight_layout()
    print("\n[INFO] Pie chart generated — look for the popup window on your desktop.")
    plt.show()


def process_and_analyze(batch_size: int, model_name: str, profile_handle: str) -> list:
    """
    Runs BitNet on unprocessed tweets for the given profile.
    FastText taxonomy is used only if USE_FASTTEXT=True in fasttext_client.py;
    otherwise BitNet's taxonomy is used (correct for all content types).
    """
    db.init_db()
    tweets = db.get_unprocessed_tweets(limit=batch_size, profile_handle=profile_handle)
    print(f"\nFound {len(tweets)} unprocessed tweets for @{profile_handle}. Beginning analysis...\n")

    ok, failed    = 0, 0
    all_categories = []

    for t in tweets:
        try:
            sentiment, entities, bitnet_taxonomy = analyze_tweet(
                t["text"], tweet_date=t.get("created_at")
            )

            fasttext_taxonomy = classify_taxonomy(t["text"])
            taxonomy = fasttext_taxonomy if fasttext_taxonomy is not None else bitnet_taxonomy

            db.save_analysis(t["tweet_id"], sentiment, entities, taxonomy, model_name)
            ok += 1

            for tax in taxonomy:
                top = tax.get("top_level", "")
                if top and top not in ("Uncategorized", "Arts & Entertainment"):
                    all_categories.append(top)

            tax_str = ", ".join(x["top_level"] for x in taxonomy)
            print("=" * 70)
            print(f"TWEET [{t['tweet_id']}]:\n{t['text']}")
            print("-" * 70)
            print(f"Sentiment : {sentiment:+d}")
            print(f"Taxonomy  : {tax_str}")
            print(f"Persons   : {entities.get('persons') or []}")
            print(f"Orgs      : {entities.get('organizations') or []}")
            print(f"Locations : {entities.get('locations') or []}")
            print("=" * 70 + "\n")

        except Exception as e:
            failed += 1
            print(f"[{t['tweet_id']}] FAILED: {e}\n")

    print(f"Analysis done. {ok} successful, {failed} failed.")
    return all_categories

def get_all_categories_for_profile(profile_handle: str) -> list:
    """Fetches ALL taxonomy categories for a profile from the database, including previously processed ones."""
    categories = []
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT tx.top_level FROM taxonomy tx
               JOIN tweets t ON tx.tweet_id = t.tweet_id
               WHERE LOWER(t.profile_handle) = LOWER(?)""",
            (profile_handle,)
        ).fetchall()
        for r in rows:
            top = r["top_level"]
            if top and top not in ("Uncategorized", "Arts & Entertainment"):
                categories.append(top)
    return categories


def main():
    parser = argparse.ArgumentParser(
        description="Twitter Profile Analyzer — Scrape → BitNet → Pie Chart"
    )
    parser.add_argument("profile_url",  nargs="?", help="e.g. https://x.com/elonmusk")
    parser.add_argument("--count",       type=int,  default=DEFAULT_TWEET_TARGET)
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping, re-analyze existing tweets")
    parser.add_argument("--headed",      action="store_true", help="Show browser window while scraping (default: browser is always visible)")
    args = parser.parse_args()

    print("=" * 60)
    print(" TWITTER PROFILE ANALYZER ".center(60))
    print("=" * 60)

    profile_url = args.profile_url
    if not profile_url and not args.skip_scrape:
        profile_url = input("\nEnter Twitter Profile URL (e.g., https://x.com/elonmusk): ").strip()

    profile_name = profile_url.rstrip("/").split("/")[-1] if profile_url else "unknown"

    # ── Phase 1: Scrape ───────────────────────────────────────────────────────
    if not args.skip_scrape and profile_url:
        print(f"\n[PHASE 1] Scraping up to {args.count} tweets from @{profile_name}...")
        print("  A browser window will open — this is normal. Do not close it.")
        n_scraped = scrape_profile(
            profile_url,
            target_count=args.count,
            profile_dir=PROFILE_DIR,
            headless=False,   # always headed: X serves real content in visible browser
        )
        print(f"  Successfully scraped {n_scraped} tweets.")
    else:
        print("\n[PHASE 1] Skipping scrape — analyzing existing tweets in DB.")

    # ── Phase 2: Analyze ──────────────────────────────────────────────────────
    print("\n[PHASE 2] Running BitNet analysis (sentiment + entities + taxonomy)...")
    process_and_analyze(
        batch_size=args.count,
        model_name=BITNET_MODEL_NAME,
        profile_handle=profile_name,
    )

    # ── Phase 3: Pie chart ────────────────────────────────────────────────────
    print("\n[PHASE 3] Rendering taxonomy pie chart...")
    # Fetch from DB so it always has data, even if 0 new tweets were processed
    all_categories = get_all_categories_for_profile(profile_name)
    counts = Counter(all_categories)
    show_pie_chart(counts, profile_name)


    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n--- DATABASE STATUS ---")
    print(db.stats(profile_handle=profile_name))
    print("=" * 60)
    print("  Run 'python report.py' for full sentiment / entity breakdown.")
    print("=" * 60)


if __name__ == "__main__":
    main()