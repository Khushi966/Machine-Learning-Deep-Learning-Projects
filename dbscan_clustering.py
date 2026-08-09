
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters_}')
print(f'Estimated number of noise points: {n_noise_}')
print(f'Silhouette Coefficient: {metrics.silhouette_score(X, labels)}')

# Plot result
plt.figure(figsize=(10, 8))

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    # Plot the core samples
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=10,
    )

    # Plot the non-core samples
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        'o',
        markerfacecolor=tuple(col),
        markeredgecolor='k',
        markersize=5,
    )

plt.title(f'Estimated number of clusters: {n_clusters_}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

def dbscan_parameter_test(X, eps_values, min_samples_values):
    """Test DBSCAN with different parameter values."""
    fig, axes = plt.subplots(len(eps_values), len(min_samples_values), figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, eps in enumerate(eps_values):
        for j, min_samples in enumerate(min_samples_values):
            
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_

            # Plot the result
            ax = axes[i, j]
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]

                class_member_mask = labels == k
                ax.plot(X[class_member_mask, 0], X[class_member_mask, 1], 'o',
                        markerfacecolor=tuple(col),
                        markeredgecolor='k',
                        markersize=3)

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            ax.set_title(f'eps={eps}, min_s={min_samples}\nClusters: {n_clusters_}, Noise: {n_noise_}')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

eps_values = [0.1, 0.2, 0.3, 0.4]
min_samples_values = [5, 10, 15, 20]
dbscan_parameter_test(X, eps_values, min_samples_values)
