
# coding: utf-8

# In[1]:

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.stats import norm
from sklearn.utils import shuffle
from math import sqrt
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

# In[2]:

class CW(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.90, n_iter=5, shuffle=True, random_state=100):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.phi = norm.cdf(self.eta)**(-1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_samples, n_classes = y.shape
        self.mu = np.ones(n_features)
        self.sigma = np.diag([1.0] * n_features)

        for epoch in range(self.n_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)

            for i in range(n_samples):
                self._update(X[i:i + 1], y[i:i + 1])

    def _update(self, X, y):
        m = y * (np.dot(X, self.mu.T))
        v = np.dot(np.dot(X, self.sigma), X.T)
        gamma = (- (1 + 2 * self.phi * m) + sqrt((1 + 2 * self.phi * m) ** 2 - 8 * self.phi * (m - self.phi * v))) / (4 * self.phi * v)
        alpha = max(0, gamma)
        self.mu = self.mu + alpha * y * np.dot(self.sigma, X.T).T
        self.sigma = (self.sigma ** (-1) + 2 * alpha * self.phi * np.diag(X)) ** (-1)

    def predict(self, X):
        return np.sign(np.dot(X, self.mu))


# In[4]:




# In[5]:

N =200

# 訓練データ生成
dat = make_classification(n_samples=N, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, shuffle=True, random_state=200)
X = dat[0]
y = np.array([-1 if d == 0 else 1 for d in dat[1]])
y = y.reshape(N, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

cw = CW(eta=0.90, n_iter=5, shuffle=True, random_state=100)
cw.fit(X_train, y_train)

y_pred = cw.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
target_names = ['0', '1']
print(classification_report(y_test, y_pred, target_names=target_names))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('r', 'b', 'g', 'y', 'm')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # 最小値, 最大値からエリアの領域を割り出す
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # resolutionの間隔で区切った領域を定義
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidth=1, marker='o',
            s=55, label='test set')

X_combined = np.vstack((X_train, X_test)) # 縦に連結
y_combined = np.hstack((y_train, y_test)) # 横に連結
plot_decision_regions(X=X_combined,
                        y=y_combined,
                        classifier=ppn,
                        test_idx=range(105, 150))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.show()


# ## Confidence Weighted Learning
#
# \begin{align}
#     \textbf{$\mu$}_{i + 1} &= \textbf{$\mu$}_i + \alpha y_i \Sigma_i \mathbf{x}_i \\
#     \Sigma_{i + 1} &= \Sigma_i - \Sigma_i \mathbf{x}_i \frac{2 \alpha \phi}{1 + 2 \alpha \phi \mathbf{x}^{\mathrm{T}}_i \Sigma_i \mathbf{x}_i} \mathbf{x}^{\mathrm{T}}_i \Sigma_i \\
#     \alpha &=
# \end{align}
