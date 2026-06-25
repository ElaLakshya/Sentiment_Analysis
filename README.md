# Twitter/X Intelligence Pipeline

A fully local, end-to-end AI pipeline for analyzing Twitter/X profiles. It scrapes tweets, scores sentiment, extracts named entities, classifies topics using the Google Product Taxonomy, and generates visualizations — no cloud APIs, no data leaving your machine.

---

## Architecture

```
Twitter/X Profile
      │
      ▼
 scraper.py          ← CloakBrowser scrolls the profile, saves raw tweets
      │
      ▼
 twitter_intel.db    ← SQLite stores everything; no tweet is ever processed twice
      │
      ▼
 ┌────┴─────────────────────────┐
 │                              │
 BitNet (llama-server)     FastText (.bin)
 Sentiment: -1 / 0 / +1    Google Taxonomy classification
 Entities: Person /         5,500+ categories
 Org / Location / Date      
 └────┬─────────────────────────┘
      │
      ▼
 Pie Chart + report.py
```

**Why two models?**
BitNet (1.58b) excels at contextual reasoning — understanding tone, sarcasm, and extracting specific names — but cannot reliably track 5,500+ taxonomy categories simultaneously. FastText uses hierarchical word vectors and is purpose-built for high-speed multi-label classification at that scale. Each model does what it is best at.

---

## Project Files

| File | Purpose |
|------|---------|
| `main.py` | Main entry point — scrape → analyze → pie chart |
| `config.py` | Central settings (DB path, BitNet URL, browser profile dir) |
| `db.py` | SQLite schema and all read/write helpers |
| `scraper.py` | CloakBrowser engine that scrolls a profile and saves raw tweets |
| `bitnet_client.py` | BitNet client for sentiment and entity extraction, with JSON auto-repair |
| `fasttext_client.py` | FastText client for Google taxonomy classification |
| `report.py` | CLI tool for exploring the database |
| `import_cookies.py` | Injects real browser cookies to bypass X's bot detection |
| `test_bitnet.py` | Sanity check for the BitNet server connection |
| `train_fasttext.py` | Trains a new `taxonomy_model.bin` from `train.txt` |

---

## One-Time Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Start the BitNet server

Launch in its own terminal and keep it running for the entire session.

```powershell
& "C:\path\to\BitNet\build\bin\Release\llama-server.exe" --model "C:\path\to\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf" --host 127.0.0.1 --port 8080 -c 4096 -b 32 -ub 16 --threads 4
```

> **The `-ub 16` flag is required.** BitNet uses experimental ternary math kernels that overflow at the default batch size, producing a NaN that crashes the server silently on the first inference request. `-ub 16` forces micro-batching that keeps the kernel within safe bounds.

Verify it is running (in a second terminal):

```powershell
curl http://127.0.0.1:8080/v1/models
```

### 3. Authenticate with X (one-time, repeat when session expires ~30 days)

X blocks headless browsers aggressively. The most reliable approach is to inject a real logged-in session rather than automating the login screen.

1. Log into **x.com** in your normal Chrome or Edge browser.
2. Install the **J2TEAM Cookies** extension from the Chrome Web Store.
3. While on x.com, open J2TEAM Cookies → Export → **leave the password field blank** → save as `cookie_export.json` in the project folder.
4. Run:

```powershell
python import_cookies.py cookie_export.json
```

A browser window briefly opens to confirm you are logged in, then closes. The session is saved to `./x_profile` and reused automatically on every subsequent run.

### 4. Build the FastText taxonomy model

The compiled `.bin` model is not included in the repository. You must train it once from your labeled data.

```powershell
# After you have a train.txt file with labeled examples:
python train_fasttext.py
```

Training data format (one example per line):
```
__label__Electronics  Just switched to the new M3 MacBook Pro, the battery life is insane.
__label__Media  This documentary on Netflix about deep sea creatures is mind-blowing.
```

See `generate_data_mistral.py` for generating synthetic training data via the Mistral API, and `cleanup_script.py` for scrubbing the output before training.

---

## Running the Pipeline

```powershell
python main.py https://x.com/USERNAME --count 100
```

What happens:

1. **Scrape** — A visible browser opens, navigates to the profile, and collects up to 100 tweets authored by the target user. Retweets and suggested content are filtered out automatically.
2. **Analyze** — Each tweet is sent through FastText (taxonomy) and BitNet (sentiment + entities). Results are written to `twitter_intel.db`.
3. **Visualize** — A pie chart of taxonomy categories appears as a popup window.

### Flags

| Flag | Effect |
|------|--------|
| `--count 50` | Scrape a specific number of tweets (default: 100) |
| `--skip-scrape` | Skip the browser; analyze unprocessed tweets already in the database |

---

## Viewing Reports

```powershell
python report.py                   # Summary: sentiment totals, top categories, top entities
python report.py --sentiment       # Sentiment bar chart breakdown
python report.py --taxonomy        # Google taxonomy categories (top-level + sub-level)
python report.py --entities        # All persons, organizations, locations, and dates
python report.py --timeline        # Day-by-day sentiment trajectory
python report.py --positive        # 10 most recent positive tweets
python report.py --negative        # 10 most recent negative tweets
python report.py --search "AI"     # All tweets mentioning a keyword
python report.py --export          # Export full database to results.csv
python report.py --all             # Run every report
```

---

## Database Layout

File: `twitter_intel.db` — open with DB Browser for SQLite or query directly.

| Table | Contents |
|-------|---------|
| `tweets` | Raw tweet data: id, handle, text, url, timestamps, processed flag |
| `sentiment` | One row per tweet, score in `{-1, 0, 1}` |
| `entities` | Named entities: type in `{person, organization, location, date}` |
| `taxonomy` | Google taxonomy: top-level category + optional sub-level |

---

## Scraping Multiple Profiles

Tweets are tagged with `profile_handle` so multiple profiles coexist in the same database without conflict. Simply run the pipeline with a different URL and the new profile is added alongside the existing data.

To wipe the database and start fresh:

```powershell
del twitter_intel.db
```

---

## How the AI Works

### Sentiment (BitNet)

BitNet judges the **author's emotional tone**, not the subject matter of the tweet.

| Score | Meaning | Example |
|-------|---------|---------|
| `+1` | Happy, excited, approving, grateful | *"This launch is incredible."* |
| `0` | Informational, factual, ambiguous | *"The Fed raised rates by 25bps."* |
| `-1` | Angry, critical, frustrated, sarcastic | *"This policy is an absolute disaster."* |

Temperature is set to `0.0` (greedy decoding) for deterministic output and to avoid NaN crashes in BitNet's experimental sampling code. A JSON auto-repair function handles truncated output when the model runs out of its token budget mid-generation.

### Taxonomy (FastText)

FastText maps each tweet to 1–3 categories from the Google Product Taxonomy (5,500+ nodes) using hierarchical word vectors. Classification happens in milliseconds per tweet. The model is trained locally on labeled data; see the training pipeline section above.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `WinError 10061` connection refused | BitNet server is not running or crashed | Check the server terminal; restart with `-ub 16` |
| Server crashes silently on first request | Missing `-ub 16` — ternary kernel NaN segfault | Add `-ub 16` to the server launch command |
| Scraped 0 tweets | Session expired or X blocked the browser | Re-run `import_cookies.py` with a fresh export from x.com |
| "No new tweets" messages during scraping | Normal — X's feed loads lazily; scraper retries up to 8 times | Wait; if it hits 8/8 and returns 0, session has expired |
| Taxonomy returns `Arts & Entertainment` for everything | `taxonomy_model.bin` not found or `USE_FASTTEXT = False` | Ensure the `.bin` is in the project folder and `USE_FASTTEXT = True` in `fasttext_client.py` |
| Wrong tweets in analysis | Old data from before anti-hallucination filter | `del twitter_intel.db` and re-run |
| Schema error on re-run | DB schema changed since last run | `del twitter_intel.db` and re-run |