"""
Alternative to login_session.py: import cookies from a normal, manually
logged-in browser session instead of automating X's login screen - which is
the most heavily defended part of the site, and the most likely place a
stealth browser gets flagged, blocked, or hangs.

How to get the cookie export:
1. In your regular Edge/Chrome (no automation at all), log into x.com as
   you normally would.
2. Install a cookie export extension - "Cookie-Editor" is a common one,
   search for it in your browser's extension store.
3. While on x.com (logged in), open Cookie-Editor -> Export -> Export as
   JSON. Save that to a file, e.g. cookie_export.json, in this folder.

Then run:
    python import_cookies.py cookie_export.json

This parser tolerates a few different export shapes (plain list of cookie
objects, a dict wrapping that list, a JSON-string-encoded array, or a list
of raw "name=value; Domain=...; Secure" header strings) since different
extensions/versions format the export differently.
"""

import json
import sys

from cloakbrowser import launch_persistent_context

from config import PROFILE_DIR

SAME_SITE_MAP = {
    "no_restriction": "None",
    "lax": "Lax",
    "strict": "Strict",
    "unspecified": "Lax",
    "none": "None",
}


def _parse_cookie_header_string(s: str):
    """Handle 'name=value; Domain=.x.com; Path=/; Secure; HttpOnly' style entries."""
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if not parts or "=" not in parts[0]:
        return None
    name, value = parts[0].split("=", 1)
    cookie = {
        "name": name.strip(),
        "value": value.strip(),
        "domain": "x.com",
        "path": "/",
        "session": True,
        "secure": False,
        "httpOnly": False,
        "sameSite": "unspecified",
    }
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "domain":
                cookie["domain"] = v
            elif k == "path":
                cookie["path"] = v
            elif k in ("expires", "max-age"):
                cookie["session"] = False
                cookie["expirationDate"] = 9999999999  # far future; good enough for scraping use
            elif k == "samesite":
                cookie["sameSite"] = v.lower()
        elif part.lower() == "secure":
            cookie["secure"] = True
        elif part.lower() == "httponly":
            cookie["httpOnly"] = True
    return cookie


def _load_raw_cookies(export_path: str):
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Some exports double-encode: the file's top-level JSON value is itself
    # a JSON string containing the real array.
    if isinstance(data, str):
        data = json.loads(data)

    # Some export formats wrap the array under a key, e.g. {"cookies": [...]}
    if isinstance(data, dict):
        # Flat {cookieName: cookieValue} mapping - no per-cookie metadata,
        # so fall back to sensible defaults for domain/path/secure/etc.
        if data and all(isinstance(v, str) for v in data.values()):
            return [
                {
                    "name": k,
                    "value": v,
                    "domain": "x.com",
                    "path": "/",
                    "secure": True,
                    "httpOnly": False,
                    "session": True,
                }
                for k, v in data.items()
            ]

        list_value = None
        for key in ("cookies", "data", "result", "items"):
            if isinstance(data.get(key), list):
                list_value = data[key]
                break
        if list_value is None:
            # Generic fallback: grab the first list-valued entry, whatever it's called.
            for v in data.values():
                if isinstance(v, list):
                    list_value = v
                    break
        if list_value is not None:
            data = list_value

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of cookies after unwrapping, got {type(data).__name__}")

    normalized = []
    for item in data:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            # Could be a JSON-encoded object-as-string, or a raw header string.
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    normalized.append(parsed)
                    continue
            except json.JSONDecodeError:
                pass
            parsed = _parse_cookie_header_string(item)
            if parsed:
                normalized.append(parsed)
    return normalized


def _convert_cookie(c: dict) -> dict:
    return {
        "name": c["name"],
        "value": c["value"],
        "domain": c.get("domain", "x.com"),
        "path": c.get("path", "/"),
        "expires": -1 if c.get("session", False) else c.get("expirationDate", -1),
        "httpOnly": c.get("httpOnly", False),
        "secure": c.get("secure", False),
        "sameSite": SAME_SITE_MAP.get(str(c.get("sameSite", "unspecified")).lower(), "Lax"),
    }


def import_cookies(export_path: str, profile_dir: str = PROFILE_DIR):
    raw_cookies = _load_raw_cookies(export_path)

    cookies = [
        _convert_cookie(c) for c in raw_cookies
        if "x.com" in c.get("domain", "") and c.get("name") and c.get("value")
    ]
    if not cookies:
        print("No usable x.com cookies found - check you exported while actually on x.com.")
        print(f"(Parsed {len(raw_cookies)} raw entries, 0 matched after filtering.)")
        return

    context = launch_persistent_context(profile_dir, headless=False)
    context.add_cookies(cookies)

    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://x.com/home", wait_until="domcontentloaded")

    input(
        f"\nImported {len(cookies)} cookies. Check the browser window - if you see your "
        "home timeline, it worked. Press Enter to save and close...\n"
    )

    context.close()
    print(f"Profile saved to {profile_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python import_cookies.py <cookie_export.json>")
        sys.exit(1)
    import_cookies(sys.argv[1])
