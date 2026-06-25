import json
import re
import requests
from config import BITNET_MODEL_NAME, BITNET_BASE_URL

# ── Google Product Taxonomy - top-level categories ────────────────────────────
GOOGLE_TAXONOMY_TOP = [
    "Animals & Pet Supplies", "Apparel & Accessories", "Arts & Entertainment",
    "Baby & Toddler", "Business & Industrial", "Cameras & Optics", "Electronics",
    "Food, Beverages & Tobacco", "Furniture", "Hardware", "Health & Beauty",
    "Home & Garden", "Luggage & Bags", "Mature", "Media", "Office Supplies",
    "Religious & Ceremonial", "Software", "Sporting Goods", "Toys & Games",
    "Vehicles & Parts",
]

TAXONOMY_LIST_STR = "\n".join(f"- {c}" for c in GOOGLE_TAXONOMY_TOP)

SYSTEM_PROMPT = f"""You are a precise data-extraction engine. For the given text, return ONLY a \
single valid JSON object - no markdown, no commentary.

RULES:
1. sentiment: 1 (happy, excited, approving), 0 (informational, neutral), or -1 (angry, critical, frustrated).
2. entities: Extract explicit names. DO NOT put organizations in the "persons" list. If none found, use [].
   - If the text uses relative dates like "today" or "now", output the provided Tweet Date.
3. taxonomy: Pick 1-2 categories from the ALLOWED list below.

ALLOWED TAXONOMY:
{TAXONOMY_LIST_STR}

=== EXAMPLE INPUT ===
Tweet Date: 2024-05-10
Text: "Loved seeing the new SpaceX rocket launch in Texas today with Elon! The engineering is mind-blowing."

=== EXAMPLE OUTPUT ===
{{
  "sentiment": 1,
  "entities": {{
    "persons": ["Elon"],
    "organizations": ["SpaceX"],
    "locations": ["Texas"],
    "dates": ["2024-05-10"]
  }},
  "taxonomy": [
    {{
      "top_level": "Vehicles & Parts",
      "sub_level": "Spacecraft"
    }}
  ]
}}
=== END EXAMPLE ===
"""


def _repair_and_parse_json(raw: str) -> dict:
    """
    Cleans and parses model output that may be malformed in these ways:
    1. Wrapped in markdown fences (```json ... ```)
    2. Has trailing model stop tokens (<|end|>, <|assistant|>, etc.)
    3. Missing closing braces/brackets (model ran out of tokens mid-generation)
    """
    raw = raw.strip()

    # Strip markdown fences
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```.*$", "", raw, flags=re.DOTALL).strip()

    # Strip model stop tokens and everything after them
    raw = re.sub(r"<\|[^>]+\|>.*$", "", raw, flags=re.DOTALL).strip()

    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found: {raw[:200]!r}")
    raw = raw[start:]

    # Try parsing as-is first (happy path)
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw)
        return parsed
    except json.JSONDecodeError:
        pass

    # Repair: count unclosed braces and brackets, append missing closers
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape_next = False
    for ch in raw:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1

    suffix = "]" * max(depth_bracket, 0) + "}" * max(depth_brace, 0)
    repaired = raw + suffix

    try:
        parsed, _ = json.JSONDecoder().raw_decode(repaired)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON even after repair: {e} | Raw: {raw[:300]!r}")


def _validate_taxonomy(entries: list) -> list:
    valid = []
    for e in (entries or []):
        if not isinstance(e, dict):
            continue
        top = str(e.get("top_level") or "").strip()
        if top in GOOGLE_TAXONOMY_TOP:
            valid.append({
                "top_level": top,
                "sub_level": str(e["sub_level"]) if e.get("sub_level") else None,
            })
    return valid or [{"top_level": "Arts & Entertainment", "sub_level": None}]


def _parse_result(content: str):
    parsed    = _repair_and_parse_json(content)
    sentiment = parsed.get("sentiment", 0)
    if sentiment not in (-1, 0, 1):
        sentiment = 0
    ents = parsed.get("entities") or {}
    for key in ("persons", "organizations", "locations", "dates"):
        ents[key] = [str(v) for v in (ents.get(key) or [])]
    taxonomy = _validate_taxonomy(parsed.get("taxonomy") or [])
    return sentiment, ents, taxonomy


def _call_completion(tweet_text: str, tweet_date: str, timeout: int) -> str:
    """Uses /completion directly. The Chat endpoint causes SegFaults in
    experimental BitNet llama.cpp forks."""
    user_input = f"Tweet Date: {tweet_date}\nText: {tweet_text}" if tweet_date else f"Text: {tweet_text}"
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_input}\n"
        f"<|assistant|>\n"
    )
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": 500,
    }
    resp = requests.post(f"{BITNET_BASE_URL}/completion", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["content"]


def analyze_tweet(tweet_text: str, tweet_date: str = None, retries: int = 2, timeout: int = 90):
    """Returns (sentiment: int, entities: dict, taxonomy: list[dict])"""
    last_err = None
    for _ in range(retries + 1):
        try:
            content = _call_completion(tweet_text, tweet_date, timeout)
            return _parse_result(content)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"BitNet failed after {retries+1} attempts: {last_err}")
