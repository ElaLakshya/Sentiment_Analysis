import os

# --- Database ---
DB_PATH = os.environ.get("TWITTER_INTEL_DB", "twitter_intel.db")

# --- BitNet (llama-server, OpenAI-compatible) ---
# This must point at the llama-server process you start from your existing
# bitnet.cpp build (see README.md for the launch command).
BITNET_BASE_URL = os.environ.get("BITNET_BASE_URL", "http://127.0.0.1:8080")
BITNET_CHAT_URL = f"{BITNET_BASE_URL}/v1/chat/completions"
BITNET_MODEL_NAME = os.environ.get("BITNET_MODEL_NAME", "bitnet-b1.58-2B-4T")

# --- Scraper / session ---
# launch_persistent_context() stores cookies/localStorage/cache in this
# directory, like a real Chrome profile - this avoids the incognito-context
# penalty some sites (X included) apply to throwaway sessions.
PROFILE_DIR = os.environ.get("X_PROFILE_DIR", "./x_profile")
DEFAULT_TWEET_TARGET = 100
