import matplotlib.pyplot as plt
import numpy as np
data_size = [(8192**2,16,"Hierarchical"), (8192**2,16,"Non-hierarchical"), (8192**2,24,"Hierarchical"), (8192**2,24,"Non-hierarchical")]
time_data = [
    [17.356520,17.287745,17.353395],  
    [17.147599,17.092598,17.125140],
    [17.373929,17.339045,17.365756],
    [17.095019,17.124894,17.249463]
]
plt.figure(figsize=(10, 6))
plt.boxplot(time_data)
positions = np.arange(1, len(data_size) + 1)
plt.xticks(positions, [f"{size[0]}, Px={size[1]},{size[2]}" for size in data_size], rotation=1.05)
plt.xlabel('(N, #processes)')
plt.ylabel('Time (seconds)')
plt.title('Time taken for each data size per stencil configuration')
plt.grid(True)
plt.tight_layout()
plt.show()

