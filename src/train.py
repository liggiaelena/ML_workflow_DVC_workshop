import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Paths
DATA_DIR = "data/processed"
MODEL_PATH = "model.pt"
METRICS_PATH = "metrics.json"

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
}


class SimpleCNN(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.activation = ACTIVATIONS[activation]
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(self.activation(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    EPOCHS = params["epochs"]
    LR = params["lr"]
    BATCH_SIZE = params["batch_size"]
    ACTIVATION = params.get("activation", "relu")
    OPTIMIZER = params.get("optimizer", "adam")
    MOMENTUM = params.get("momentum", 0.9)

    train_images, train_labels = torch.load(os.path.join(DATA_DIR, "train.pt"))
    test_images, test_labels = torch.load(os.path.join(DATA_DIR, "test.pt"))

    model = SimpleCNN(activation=ACTIVATION)
    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LR)
    elif OPTIMIZER == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=LR)

    grad_stats = {}

    model.train()
    for epoch in range(EPOCHS):
        for i in range(0, len(train_images), BATCH_SIZE):
            x_batch = train_images[i:i + BATCH_SIZE]
            y_batch = train_labels[i:i + BATCH_SIZE]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # --- Challenge 6.3: Forward/Backward Pass Instrumentation ---
            # Runs once on the very first batch of the first epoch
            if epoch == 0 and i == 0:
                conv_weight_norm_before = model.conv.weight.data.norm().item()

                print("\n--- Forward Pass (first batch) ---")
                print(f"Output logits (first 5 samples):\n{outputs[:5].detach()}")
                print(f"Loss: {loss.item():.4f}")

                loss.backward()

                conv_grad_norm = model.conv.weight.grad.norm().item()
                fc_grad_norm = model.fc.weight.grad.norm().item()

                print("\n--- Backward Pass (first batch) ---")
                print(f"Conv weight grad norm : {conv_grad_norm:.6f}")
                print(f"FC   weight grad norm : {fc_grad_norm:.6f}")

                optimizer.step()

                conv_weight_norm_after = model.conv.weight.data.norm().item()
                print(f"\nConv weight norm before update: {conv_weight_norm_before:.6f}")
                print(f"Conv weight norm after  update: {conv_weight_norm_after:.6f}")
                print("----------------------------------\n")

                grad_stats = {
                    "conv_grad_norm": round(conv_grad_norm, 6),
                    "fc_grad_norm": round(fc_grad_norm, 6),
                    "conv_weight_norm_before": round(conv_weight_norm_before, 6),
                    "conv_weight_norm_after": round(conv_weight_norm_after, 6),
                    "first_batch_loss": round(loss.item(), 4),
                }
            else:
                loss.backward()
                optimizer.step()
            # --- End instrumentation ---

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        outputs = model(test_images)
        _, predicted = torch.max(outputs, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

    metrics = {"accuracy": round(accuracy, 4)}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open("grad_stats.json", "w", encoding="utf-8") as f:
        json.dump(grad_stats, f, indent=4)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print("Gradient stats saved to grad_stats.json")


if __name__ == "__main__":
    main()
