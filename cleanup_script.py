import os

def clean_data():
    # 1. Generate a "Master List" of mathematically perfect labels
    valid_labels = set()
    with open("taxonomy.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Convert "Apparel & Accessories" to "__label__Apparel_&_Accessories"
                perfect_label = "__label__" + line.replace(" ", "_")
                valid_labels.add(perfect_label)
                
    print(f"Loaded {len(valid_labels)} perfect labels from taxonomy.")

    # 2. Scan train.txt and ONLY keep lines with a perfect label
    good_lines = 0
    bad_lines = 0
    
    with open("train.txt", "r", encoding="utf-8") as f_in, open("train_cleaned.txt", "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # Split the line into [Label, Sentence]
            parts = line.split(" ", 1)
            
            # If the label exists in our Master List, save it!
            if len(parts) == 2 and parts[0] in valid_labels:
                f_out.write(line + "\n")
                good_lines += 1
            else:
                # This label has a typo or is formatted wrong! Throw it away.
                bad_lines += 1

    print("\nCleanup Complete!")
    print(f"Kept: {good_lines} perfectly formatted lines.")
    print(f"Removed: {bad_lines} lines with typos or bad formatting.")
    print("\nPlease delete your old 'train.txt' and rename 'train_cleaned.txt' to 'train.txt'.")

if __name__ == "__main__":
    clean_data()