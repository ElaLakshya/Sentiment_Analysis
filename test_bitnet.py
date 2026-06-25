"""
Run this after starting llama-server to confirm BitNet is responding correctly.
Watch the llama-server terminal while this runs - any crash/error will show there.
"""
from bitnet_client import analyze_tweet
from fasttext_client import classify_taxonomy

SAMPLE = (
    "My new laptop is so fast, the screen is beautiful, and the software is amazing!"
)

if __name__ == "__main__":
    print("Sending test tweet to BitNet & FastText...")
    print(f"Tweet: {SAMPLE}\n")
    
    # 1. BitNet Extraction
    sentiment, entities = analyze_tweet(SAMPLE, tweet_date="2023-12-01")
    
    # 2. FastText Classification
    taxonomy = classify_taxonomy(SAMPLE)
    
    print(f"Sentiment : {sentiment:+d}")
    print(f"Persons   : {entities.get('persons', [])}")
    print(f"Orgs      : {entities.get('organizations', [])}")
    print(f"Locations : {entities.get('locations', [])}")
    print(f"Dates     : {entities.get('dates', [])}")
    print(f"Taxonomy  : {[t['top_level'] for t in taxonomy]}")
    print("\nSuccess!")