import numpy as np
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

N =200

# 訓練データ生成
dat = make_classification(n_samples=N, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.7, 0.3], flip_y=0.01, shuffle=True, random_state=200)
X = dat[0]
y = dat[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

ppn = Perceptron(n_iter=1, eta0=0.1)
ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test)

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

#X_combined = np.vstack((X_train, X_test)) # 縦に連結
#y_combined = np.hstack((y_train, y_test)) # 横に連結
plot_decision_regions(X=X,
                        y=y,
                        classifier=ppn,
                        test_idx=range(105, 150))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.show()
