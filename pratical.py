import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        self.X = X
        self.n_samples = len(X)
        self.visited = np.zeros(self.n_samples, dtype=bool)
        self.labels = np.zeros(self.n_samples, dtype=int)
        self.current_label = 0

        for i in range(self.n_samples):
            if not self.visited[i]:
                self.visited[i] = True
                neighbors = self.range_query(i)
                if len(neighbors) < self.min_samples:
                    self.labels[i] = -1
                else:
                    self.current_label += 1
                    self.expand_cluster(i, neighbors, self.current_label)

        return self.labels

    def expand_cluster(self, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not self.visited[neighbor_idx]:
                self.visited[neighbor_idx] = True
                new_neighbors = self.range_query(neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            if self.labels[neighbor_idx] == 0:
                self.labels[neighbor_idx] = cluster_id
            i += 1

    def range_query(self, point_idx):
        neighbors = []
        for i in range(self.n_samples):
            if np.linalg.norm(self.X[point_idx] - self.X[i]) < self.eps:
                neighbors.append(i)
        return neighbors


class OutlierDetectionIQR:
    def __init__(self):
        pass

    def detect_outliers(self, X):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        return outliers


class OutlierDetectionZScore:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, X):
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        outliers = z_scores > self.threshold
        return np.any(outliers, axis=1)


class OutlierDetectionLOF:
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def detect_outliers(self, X):
        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors)
        outliers = lof.fit_predict(X)
        return outliers == -1


if __name__ == "__main__":
    data = pd.read_csv('Mall_Customers.csv')
    X = data[['Age', 'Spending Score (1-100)']].values

    # DBSCAN
    dbscan = DBSCAN(eps=5, min_samples=10)
    clusters = dbscan.fit_predict(X)
    core_points = X[clusters != -1]
    noise_points = X[clusters == -1]

    # Outlier detection using IQR
    outlier_detector_iqr = OutlierDetectionIQR()
    outliers_iqr = outlier_detector_iqr.detect_outliers(X)

    # Outlier detection using Z-score
    outlier_detector_z_score = OutlierDetectionZScore(threshold=3)
    outliers_z_score = outlier_detector_z_score.detect_outliers(X)

    # Outlier detection using LOF
    outlier_detector_lof = OutlierDetectionLOF(n_neighbors=20)
    outliers_lof = outlier_detector_lof.detect_outliers(X)

    # Plotting the clusters and outliers
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 5))

    # Plot DBSCAN clusters
    plt.subplot(1, 4, 1)
    plt.scatter(core_points[:, 0], core_points[:, 1], c=clusters[clusters != -1], cmap='viridis', marker='o', label='Core Points')
    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', label='Noise Points')
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.title("DBSCAN Clustering of Mall Customers")
    plt.legend()

    # Plot outliers detected by IQR
    plt.subplot(1, 4, 2)
    plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outliers_iqr, 0], X[outliers_iqr, 1], c='red', marker='x', label='Outliers (IQR)')
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.title("Outlier Detection using IQR")
    plt.legend()

    # Plot outliers detected by Z-score
    plt.subplot(1, 4, 3)
    plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outliers_z_score, 0], X[outliers_z_score, 1], c='red', marker='x', label='Outliers (Z-score)')
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.title("Outlier Detection using Z-score")
    plt.legend()

    # Plot outliers detected by LOF
    plt.subplot(1, 4, 4)
    plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outliers_lof, 0], X[outliers_lof, 1], c='red', marker='x', label='Outliers (LOF)')
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.title("Outlier Detection using LOF")
    plt.legend()

    plt.tight_layout()
    plt.show()
