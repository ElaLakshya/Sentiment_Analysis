import fasttext

# Load your newly trained model
model = fasttext.load_model("taxonomy_model.bin")

print("Type a sentence and press Enter to see what the AI thinks it is!")
print("Type 'quit' to exit.\n")

while True:
    text = input("Tweet: ")
    if text.lower() == 'quit':
        break
        
    # Ask FastText to predict the top 3 most likely categories
    predictions, probabilities = model.predict(text, k=3)
    
    print("\nPredictions:")
    for i in range(len(predictions)):
        # Clean up the label for reading
        clean_label = predictions[i].replace("__label__", "").replace("_", " ")
        confidence = probabilities[i] * 100
        print(f"{i+1}. {clean_label} ({confidence:.1f}%)")
    print("-" * 40)