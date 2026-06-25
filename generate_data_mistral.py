import os
import time
import requests
import re

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("WARNING: MISTRAL_API_KEY environment variable not set. It might fail!")

MODEL_ID = "mistral-small-latest" 
API_URL = "https://api.mistral.ai/v1/chat/completions"

INPUT_FILE = "taxonomy.txt"
OUTPUT_FILE = "train2.txt"
TRACKING_FILE = "processed2.txt" # NEW: Absolute source of truth for progress
CHUNK_SIZE = 1 

SYSTEM_PROMPT = """You are a raw data generator. Output NOTHING except lines starting with __label__.
Do not use markdown. Do not number lines. Do not use bold headers. Do not include category titles.

Format:
__label__Category_Name Sentence text here.

Examples:
__label__Electronics_>_Computers New laptop is so fast!
__label__Apparel_&_Accessories_>_Clothing_>_Skirts Loving my new skirt!

Generate 30 sentences per category. DO NOT output conversational filler or meta-commentary.
"""

def get_already_processed():
    """Reads both the old train.txt (for past progress) and the new tracking file."""
    processed_strict_labels = set()
    processed_exact_cats = set()
    
    # Load from old train.txt (Fallback for past progress)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("__label__"):
                    label = line.split(" ")[0]
                    processed_strict_labels.add(label)
                    
    # Load from new accurate tracking file
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_exact_cats.add(line.strip())
                    
    return processed_strict_labels, processed_exact_cats

def format_label(category):
    return "__label__" + category.replace(" ", "_")

def generate_chunk(categories):
    user_message = "Generate 30 sentences for each of these categories:\n" + "\n".join(categories)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"].strip()

def main():
    strict_labels, exact_cats = get_already_processed()
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_categories = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # Filter pending categories using our new robust logic
    pending_categories = []
    for c in all_categories:
        if c in exact_cats:
            continue # We definitely processed this one
        if format_label(c) in strict_labels:
            continue # We processed it perfectly in the past
        pending_categories.append(c)
    
    print(f"Total categories in taxonomy: {len(all_categories)}")
    print(f"Already processed: {len(all_categories) - len(pending_categories)}")
    print(f"Remaining to process: {len(pending_categories)}\n")
    
    if not pending_categories:
        print("All categories have been generated! You are ready to train.")
        return
    
    for i in range(0, len(pending_categories), CHUNK_SIZE):
        chunk = pending_categories[i:i+CHUNK_SIZE]
        print(f"Processing chunk {i//CHUNK_SIZE + 1} / {(len(pending_categories)//CHUNK_SIZE) + 1}...")
        
        try:
            raw_data = generate_chunk(chunk)
            expected_label = format_label(chunk[0])
            
            valid_lines = []
            for line in raw_data.split('\n'):
                line = line.strip()
                if line.startswith("__label__") and " " in line:
                    # Strip away the AI's messy label to eliminate typos
                    clean_sentence = re.sub(r'^__label__\S+', '', line).strip()
                    
                    if len(clean_sentence) > 5:
                        # Forcefully stamp the mathematically perfect label on the front
                        valid_lines.append(f"{expected_label} {clean_sentence}")
            
            if valid_lines:
                valid_lines = valid_lines[:-1]
            
            if valid_lines:
                # 1. Save the actual generated data
                with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
                    out.write("\n".join(valid_lines) + "\n")
                
                # 2. Safely log our progress in the tracker file
                with open(TRACKING_FILE, "a", encoding="utf-8") as track:
                    for c in chunk:
                        track.write(c + "\n")
                        
                print(f"  Saved {len(valid_lines)} clean lines.")
            else:
                print("  Warning: AI generated no valid lines in this chunk.")
            
            time.sleep(2) 
        except Exception as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    main()