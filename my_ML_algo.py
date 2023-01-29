import numpy as np


def apply_PCA(raw_features, variance_percentage_kept=0.85):
    mean_column = np.mean(raw_features, axis=0)
    mean_column = np.expand_dims(mean_column, axis=0)
    mean_matrix = np.ones((raw_features.shape[0], 1))@mean_column
    centered_data = raw_features-mean_matrix
    covariance_matrix = np.dot(centered_data.T, centered_data)
    eigvals, eigvects = np.linalg.eig(covariance_matrix)
    order = np.argsort(eigvals)[::-1]
    ordered_eigvals = eigvals[order]
    power = np.sum(eigvals)

    k = 0
    variance_percentage = 0
    while variance_percentage < variance_percentage_kept:
        k += 1
        variance_percentage = np.sum(ordered_eigvals[:k])/power

    U = eigvects[order][:k]
    PCA_features = np.dot(centered_data, U.T)
    return PCA_features
