import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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


def forward_selection(X, Y, k_fold_number=5, significance_level=0.05, best_features_init=[]):
    initial_features = list(np.arange(X.shape[-1]))
    best_features = best_features_init
    remaining_features = list(set(initial_features)-set(best_features))

    classifier_improved = True
    prev_perf = 0
    can_explore = len(remaining_features) > 0
    while can_explore and classifier_improved:
        print("*****")
        print("---- best_features so far:", best_features, " ----")
        print("---- perf so far:", prev_perf, " ----")
        print("---- remaining_features so far:", remaining_features, " ----")
        print()

        best_perf = -np.infty
        best_feature_id = None
        for count, feature_id in enumerate(remaining_features):
            time_start = time.time()
            print("---- ---- ratio features explored:",
                  count/len(remaining_features))
            rnd_clf = RandomForestClassifier(verbose=0, n_jobs=-1)
            features_kept = best_features+[feature_id]
            print("---- ---- features_kept:", features_kept)
            extracted_X = X[:, features_kept]
            val_scores = cross_val_score(
                rnd_clf, extracted_X, Y, scoring="f1_macro", cv=k_fold_number)
            perf = val_scores.mean()-val_scores.std()
            print("---- ---- perf:", perf)

            if perf > best_perf:
                best_perf = perf
                best_feature_id = feature_id
            time_end = time.time()
            print("---- ---- time execution iteration:", time_end-time_start)
            print()

        if ((best_perf-prev_perf) < significance_level):
            print("STOP: small improvement")
            classifier_improved = False
            if best_perf > prev_perf:
                best_features.append(best_feature_id)
        else:
            best_features.append(best_feature_id)

        remaining_features = list(set(initial_features)-set(best_features))
        can_explore = len(remaining_features) > 0
        print("---- can_explore:", can_explore, " ----")
        print("*****")

        prev_perf = best_perf
    return best_features
