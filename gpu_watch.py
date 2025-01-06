import matplotlib.pyplot as plt
import pandas as pd

# Load GPU utilization data
data = pd.read_csv("gpu_utilization_async.csv")

# Normalize time
start_time = data["Time"].min()
data["Time"] = data["Time"] - start_time

# Plot utilization for each GPU
for gpu_id in data["GPU_ID"].unique():
    gpu_data = data[data["GPU_ID"] == gpu_id]
    plt.plot(gpu_data["Time"], gpu_data["Utilization"], label=f"GPU {gpu_id}")

plt.xlabel("Time (s)")
plt.ylabel("GPU Utilization (%)")
plt.title("GPU Utilization Over Time")
plt.legend()
plt.show()