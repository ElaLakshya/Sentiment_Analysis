import fasttext

print("--- DATASET HEALTH ---")
try:
    with open("train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Total sentences in train.txt: {len(lines)}")
except Exception as e:
    print(f"Could not read train.txt: {e}")

print("\n--- MODEL HEALTH ---")
try:
    model = fasttext.load_model("taxonomy_model.bin")
    print(f"Total Categories learned: {len(model.labels)}")
    print(f"Total Vocabulary (words learned): {len(model.words)}")
except Exception as e:
    print(f"Could not load model: {e}")