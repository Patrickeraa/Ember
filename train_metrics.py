import matplotlib.pyplot as plt
import json

# Load metrics
with open("async_metrics.json", "r") as file:
    metrics = json.load(file)

losses = metrics["losses"]
accuracies = metrics["accuracies"]

# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# Plotting Accuracy
plt.figure(figsize=(10, 5))
plt.plot(accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy Over Epochs")
plt.legend()
plt.show()