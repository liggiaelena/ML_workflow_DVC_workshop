import json
import os
import torch

# Import the model class from train.py
# IMPORTANT: Ensure that the SimpleCNN class is defined in src/train.py and is accessible here
# and also when the code is run from a Python script, the current working directory should be the project root
from train import SimpleCNN

DATA_PATH = os.path.join("data", "processed", "test.pt")
MODEL_PATH = "model.pt"
OUTPUT_PATH = "predictions.json"

# Load data
test_images, test_labels = torch.load(DATA_PATH)

# Instantiate model
model = SimpleCNN()

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    outputs = model(test_images)
    predicted = torch.argmax(outputs, dim=1)

sample_results = []
for i in range(10):
    sample_results.append({
        "index": i,
        "true_label": int(test_labels[i].item()),
        "predicted_label": int(predicted[i].item())
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sample_results, f, indent=4)

print(f"Saved predictions to {OUTPUT_PATH}")