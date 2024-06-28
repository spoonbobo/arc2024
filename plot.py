import matplotlib.pyplot as plt
import numpy as np

# Assuming you've saved the data in a file named 'training_data.txt'
data = np.loadtxt('training_data.txt', skiprows=2)

steps = data[:, 0]
loss = data[:, 1]

plt.figure(figsize=(12, 6))
plt.plot(steps, loss, label='Training Loss')
plt.title('Training Loss over Steps')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Add a trend line
z = np.polyfit(steps, loss, 1)
p = np.poly1d(z)
plt.plot(steps, p(steps), "r--", label='Trend')

plt.legend()
plt.tight_layout()
plt.show()