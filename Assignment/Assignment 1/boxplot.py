import matplotlib.pyplot as plt
import numpy as np
data_size = [(512**2, 5), (512**2, 9), (2048**2, 5), (2048**2, 9)]
time_data = [
    [0.181051,0.172100,0.161683],  
    [0.205612,0.204359,0.207448], 
    [2.241994,1.545128,2.156342],
    [3.145912,2.116054,2.408558]   
]
plt.figure(figsize=(10, 6))
plt.boxplot(time_data)
positions = np.arange(1, len(data_size) + 1)
plt.xticks(positions, [f"{size[0]}, stencil {size[1]}" for size in data_size], rotation=45)
plt.xlabel('(N, stencil)')
plt.ylabel('Time (seconds)')
plt.title('Time taken for each data size per stencil configuration')
plt.grid(True)
plt.tight_layout()
plt.show()
