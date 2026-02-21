import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.imshow(np.zeros((10, 10)))
ax.set_xticks(np.arange(0, 10, 1), minor=True)
ax.set_yticks(np.arange(0, 10, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
fig.savefig("test1.png")

fig2, ax2 = plt.subplots()
ax2.imshow(np.zeros((10, 10)))
ax2.set_xticks(np.arange(0, 10, 1), minor=True)
ax2.set_yticks(np.arange(0, 10, 1), minor=True)
ax2.grid(which="minor", color="gray", linewidth=1.5)
ax2.set_xticks([])
ax2.set_yticks([])
fig2.savefig("test2.png")
