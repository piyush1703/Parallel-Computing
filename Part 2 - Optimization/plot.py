import matplotlib.pyplot as plt
import numpy as np
data_size = [(4096**2,8,"Yes"), (4096**2,8,"No"), (4096**2, 12,"Yes"), (4096**2, 12,"No"),(8192**2, 8,"Yes"), (8192**2, 8,"No"), (8192**2, 12,"Yes"), (8192**2, 12,"No")]
time_data = [
    [7.468959, 15.763596, 27.892204, 21.452720, 38.003865],
    [7.628035, 13.634417, 27.171348, 21.758527, 35.611912],
    [7.472380, 13.844432, 27.967819, 21.146070, 35.394538],
    [7.956999, 14.112724, 28.589095, 21.867494, 31.646104],
    [29.566065, 68.929009, 106.723413, 85.634766, 89.831524],
    [31.587199, 94.175615, 113.491181, 84.627200, 96.338694],
    [32.409249, 121.097027, 101.099566, 88.027873, 93.039710],
    [32.313591, 91.172354, 106.977327, 91.703988, 94.064462]
]
plt.figure(figsize=(10, 6))
plt.boxplot(time_data)
positions = np.arange(1, len(data_size) + 1)
plt.xticks(positions, [f"{size[0]}, np = {size[1]}, {size[2]}" for size in data_size], rotation=45)
plt.xlabel('(N, processes, hierarchical)')
plt.ylabel('Time (seconds)')
plt.title('Time taken for each combination of data size, processes and hierarchical/non-hierarchical implementations')
plt.grid(True)
plt.tight_layout()

plt.savefig("plot.jpg")
plt.close()

