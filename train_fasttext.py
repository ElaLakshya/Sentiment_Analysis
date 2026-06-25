import os
import sys

try:
    import fasttext
except ImportError:
    print("Error: fasttext is not installed. Run 'pip install -r requirements.txt'")
    sys.exit(1)

def train_model():
    print("Training FastText model... This will take a minute.")

# Added loss='hs' for large taxonomies and adjusted epochs/lr
    model = fasttext.train_supervised(
        input="train.txt", 
        lr=0.5, 
        epoch=25, 
        wordNgrams=2, 
        loss='ova'
    )

    model.save_model("taxonomy_model.bin")
    print("Success! Model saved as taxonomy_model.bin")
    print("You can now run 'python main.py' to classify tweets using this model.")

if __name__ == "__main__":
    if not os.path.exists("train.txt"):
        print("Error: 'train.txt' not found!")
        print("\nPlease create 'train.txt' with your training data.")
        print("Format each line like this:")
        print("__label__Electronics __label__Software Just installed the new iOS update!")
    else:
        train_model()