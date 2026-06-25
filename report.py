"""
report.py - explore the twitter_intel database

python report.py                     # full summary
python report.py --sentiment         # score breakdown
python report.py --entities          # Task 4: persons / orgs / locations / dates
python report.py --taxonomy          # Task 5: Google taxonomy categories
python report.py --timeline          # sentiment by day
python report.py --positive          # most recent positive tweets
python report.py --negative          # most recent negative tweets
python report.py --search "keyword"  # full-text search
python report.py --export            # dump to results.csv
python report.py --all               # everything
"""

import argparse
import csv
import sqlite3
from config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _div(title=""):
    w = 65
    print(f"\n{'─'*3} {title} {'─'*(w-len(title)-5)}" if title else "─"*w)

def _bar(count, max_count, width=28):
    n = int((count / max_count) * width) if max_count else 0
    return f"{'█'*n} {count}"


# ── reports ───────────────────────────────────────────────────────────────────

def report_summary():
    db = _conn()
    total     = db.execute("SELECT COUNT(*) c FROM tweets").fetchone()["c"]
    processed = db.execute("SELECT COUNT(*) c FROM tweets WHERE processed=1").fetchone()["c"]
    handle    = db.execute("SELECT profile_handle FROM tweets LIMIT 1").fetchone()
    handle    = handle["profile_handle"] if handle else "unknown"

    pos = db.execute("SELECT COUNT(*) c FROM sentiment WHERE score= 1").fetchone()["c"]
    neu = db.execute("SELECT COUNT(*) c FROM sentiment WHERE score= 0").fetchone()["c"]
    neg = db.execute("SELECT COUNT(*) c FROM sentiment WHERE score=-1").fetchone()["c"]
    net = pos - neg

    top_tax = db.execute(
        "SELECT top_level, COUNT(*) c FROM taxonomy GROUP BY top_level ORDER BY c DESC LIMIT 4"
    ).fetchall()
    top_per = db.execute(
        "SELECT entity_value, COUNT(*) c FROM entities WHERE entity_type='person' "
        "GROUP BY entity_value ORDER BY c DESC LIMIT 4"
    ).fetchall()
    top_org = db.execute(
        "SELECT entity_value, COUNT(*) c FROM entities WHERE entity_type='organization' "
        "GROUP BY entity_value ORDER BY c DESC LIMIT 4"
    ).fetchall()

    _div("SUMMARY")
    print(f"  Profile      : @{handle}")
    print(f"  Tweets       : {total} total, {processed} analyzed")
    print(f"  Sentiment    : +{pos} positive  {neu} neutral  -{neg} negative  (net {net:+d})")
    print(f"  Top taxonomy : {', '.join(r['top_level'] for r in top_tax) or 'none'}")
    print(f"  Top persons  : {', '.join(r['entity_value'] for r in top_per) or 'none'}")
    print(f"  Top orgs     : {', '.join(r['entity_value'] for r in top_org) or 'none'}")
    db.close()


def report_sentiment():
    db = _conn()
    pos = db.execute("SELECT COUNT(*) c FROM sentiment WHERE score= 1").fetchone()["c"]
    neu = db.execute("SELECT COUNT(*) c FROM sentiment WHERE score= 0").fetchone()["c"]
    neg = db.execute("SELECT COUNT(*) c FROM sentiment WHERE score=-1").fetchone()["c"]
    total  = pos + neu + neg or 1
    max_c  = max(pos, neu, neg, 1)

    _div("SENTIMENT BREAKDOWN")
    print(f"  Positive (+1)  {_bar(pos, max_c)}  ({pos/total*100:.0f}%)")
    print(f"  Neutral   (0)  {_bar(neu, max_c)}  ({neu/total*100:.0f}%)")
    print(f"  Negative (-1)  {_bar(neg, max_c)}  ({neg/total*100:.0f}%)")
    print(f"\n  Net score: {pos-neg:+d}")
    db.close()


def report_taxonomy(limit=21):
    db = _conn()

    # Top-level breakdown
    rows = db.execute(
        "SELECT top_level, COUNT(*) c FROM taxonomy GROUP BY top_level ORDER BY c DESC LIMIT ?",
        (limit,)
    ).fetchall()
    max_c = rows[0]["c"] if rows else 1

    _div("TASK 5 – GOOGLE TAXONOMY (top-level)")
    for r in rows:
        print(f"  {r['top_level']:<32}  {_bar(r['c'], max_c)}")

    # Sub-level breakdown (where the model was specific)
    subs = db.execute(
        "SELECT sub_level, COUNT(*) c FROM taxonomy WHERE sub_level IS NOT NULL "
        "GROUP BY sub_level ORDER BY c DESC LIMIT 20"
    ).fetchall()
    if subs:
        _div("TASK 5 – GOOGLE TAXONOMY (sub-level drill-down)")
        max_s = subs[0]["c"]
        for r in subs:
            print(f"  {r['sub_level']:<48}  {_bar(r['c'], max_s)}")
    db.close()


def report_entities(limit=15):
    db = _conn()
    for etype, label in [("person","PERSONS"), ("organization","ORGANIZATIONS"),
                          ("location","LOCATIONS"), ("date","DATES")]:
        rows = db.execute(
            "SELECT entity_value, COUNT(*) c FROM entities "
            "WHERE entity_type=? GROUP BY entity_value ORDER BY c DESC LIMIT ?",
            (etype, limit)
        ).fetchall()
        _div(f"TASK 4 – {label}")
        if rows:
            max_c = rows[0]["c"]
            for r in rows:
                print(f"  {r['entity_value']:<38}  {_bar(r['c'], max_c)}")
        else:
            print("  (none found)")
    db.close()


def report_timeline():
    db = _conn()
    rows = db.execute(
        """SELECT substr(t.created_at,1,10) AS day,
                  SUM(CASE WHEN s.score= 1 THEN 1 ELSE 0 END) pos,
                  SUM(CASE WHEN s.score= 0 THEN 1 ELSE 0 END) neu,
                  SUM(CASE WHEN s.score=-1 THEN 1 ELSE 0 END) neg
           FROM tweets t JOIN sentiment s ON t.tweet_id=s.tweet_id
           WHERE t.created_at IS NOT NULL
           GROUP BY day ORDER BY day DESC LIMIT 30"""
    ).fetchall()
    _div("SENTIMENT OVER TIME  (last 30 days)")
    print(f"  {'Date':<12}  {'+':<4}  {'0':<4}  {'-':<4}  {'Net':>5}  Visual")
    _div()
    for r in rows:
        net = r["pos"] - r["neg"]
        bar = "+" * r["pos"] + "·" * r["neu"] + "-" * r["neg"]
        print(f"  {r['day']:<12}  {r['pos']:<4}  {r['neu']:<4}  {r['neg']:<4}  {net:>+5}  {bar}")
    db.close()


def report_tweets(score: int, limit=10):
    label = {1: "POSITIVE", 0: "NEUTRAL", -1: "NEGATIVE"}[score]
    db = _conn()
    rows = db.execute(
        """SELECT t.tweet_id, t.text, t.created_at, t.tweet_url
           FROM tweets t JOIN sentiment s ON t.tweet_id=s.tweet_id
           WHERE s.score=? ORDER BY t.created_at DESC LIMIT ?""",
        (score, limit)
    ).fetchall()
    _div(f"MOST RECENT {label} TWEETS (top {limit})")
    for i, r in enumerate(rows, 1):
        text = (r["text"] or "").replace("\n", " ")
        # fetch taxonomy and entities for this tweet
        tax  = db.execute("SELECT top_level FROM taxonomy WHERE tweet_id=?",
                          (r["tweet_id"],)).fetchall()
        tax_str = " | ".join(x["top_level"] for x in tax) or "—"
        print(f"\n  [{i}] {r['created_at'] or ''}  [{tax_str}]")
        print(f"  {text[:145]}{'…' if len(text)>145 else ''}")
        print(f"  {r['tweet_url']}")
    db.close()


def report_search(keyword: str):
    db = _conn()
    rows = db.execute(
        """SELECT t.tweet_id, t.text, t.created_at, t.tweet_url,
                  COALESCE(s.score, 999) score
           FROM tweets t LEFT JOIN sentiment s ON t.tweet_id=s.tweet_id
           WHERE t.text LIKE ? ORDER BY t.created_at DESC""",
        (f"%{keyword}%",)
    ).fetchall()
    _div(f"SEARCH: '{keyword}'  ({len(rows)} results)")
    for i, r in enumerate(rows, 1):
        slabel = {1:"POS", 0:"NEU", -1:"NEG"}.get(r["score"], "???")
        text   = (r["text"] or "").replace("\n", " ")
        print(f"\n  [{i}] {r['created_at'] or ''}  [{slabel}]")
        print(f"  {text[:145]}{'…' if len(text)>145 else ''}")
    db.close()


def export_csv(path="results.csv"):
    db = _conn()
    rows = db.execute(
        """SELECT
               t.tweet_id, t.profile_handle, t.created_at, t.text, t.tweet_url,
               s.score AS sentiment,
               GROUP_CONCAT(DISTINCT tx.top_level)                              AS taxonomy_top,
               GROUP_CONCAT(DISTINCT tx.sub_level)                              AS taxonomy_sub,
               GROUP_CONCAT(DISTINCT CASE WHEN e.entity_type='person'
                   THEN e.entity_value END)                                     AS persons,
               GROUP_CONCAT(DISTINCT CASE WHEN e.entity_type='organization'
                   THEN e.entity_value END)                                     AS organizations,
               GROUP_CONCAT(DISTINCT CASE WHEN e.entity_type='location'
                   THEN e.entity_value END)                                     AS locations,
               GROUP_CONCAT(DISTINCT CASE WHEN e.entity_type='date'
                   THEN e.entity_value END)                                     AS dates
           FROM tweets t
           LEFT JOIN sentiment s  ON t.tweet_id = s.tweet_id
           LEFT JOIN taxonomy  tx ON t.tweet_id = tx.tweet_id
           LEFT JOIN entities  e  ON t.tweet_id = e.tweet_id
           GROUP BY t.tweet_id
           ORDER BY t.created_at DESC"""
    ).fetchall()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])
    print(f"Exported {len(rows)} rows → {path}")
    db.close()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", action="store_true")
    parser.add_argument("--entities",  action="store_true")
    parser.add_argument("--taxonomy",  action="store_true")
    parser.add_argument("--timeline",  action="store_true")
    parser.add_argument("--positive",  action="store_true")
    parser.add_argument("--negative",  action="store_true")
    parser.add_argument("--search",    type=str, metavar="KEYWORD")
    parser.add_argument("--export",    action="store_true")
    parser.add_argument("--all",       action="store_true")
    args = parser.parse_args()

    run_all = args.all or not any(vars(args).values())

    report_summary()
    if run_all or args.sentiment: report_sentiment()
    if run_all or args.taxonomy:  report_taxonomy()
    if run_all or args.entities:  report_entities()
    if run_all or args.timeline:  report_timeline()
    if run_all or args.positive:  report_tweets(score=1)
    if run_all or args.negative:  report_tweets(score=-1)
    if args.search:               report_search(args.search)
    if run_all or args.export:    export_csv()
    print()
