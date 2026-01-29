import numpy as np
import matplotlib.pyplot as plt

# Datei laden
data = np.load("training/german_printed_chars.npz", allow_pickle=True)

X = data['X']
y = data['y']
chars = list(data['chars'])

# 25 zufällige Beispiele
indices = np.random.choice(len(X), size=25, replace=False)

# Anzeigen
fig, axes = plt.subplots(5, 5, figsize=(12, 5))

for ax, idx in zip(axes.flat, indices):
    ax.imshow(X[idx], cmap='gray')
    ax.set_title(f"'{chars[y[idx]]}'")
    ax.axis('off')

plt.tight_layout()
plt.show()