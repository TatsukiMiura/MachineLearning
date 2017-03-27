import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0, shuffle=True, random_state=200)

x0 = x[:, 0]
x1 = x[:, 1]

for (a, b) in zip(x0, x1):
    if a < b:
        plt.plot(a, b, 'ro')
    else:
        plt.plot(a, b, 'bo')

plt.show()
