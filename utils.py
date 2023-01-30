from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os


def get_distances(samples1, samples2):
    assert len(samples1.shape) == 2 and len(
        samples2.shape) == 2, "wrong shape in argument"
    assert samples1.shape[-1] == samples2.shape[-1], "not same final dimensions"

    n, m, p = samples1.shape[0], samples2.shape[0], samples1.shape[1]

    aug_samples1 = np.expand_dims(samples1, -1)  # get n,p,1
    aug_samples1 = np.matmul(aug_samples1, np.ones((n, 1, m)))  # get n,p,m
    aug_samples1 = np.transpose(aug_samples1, (0, 2, 1))  # get n,m,p

    aug_samples2 = np.expand_dims(samples2, -1)  # get m,p,1
    aug_samples2 = np.matmul(aug_samples2, np.ones((m, 1, n)))  # get m,p,n
    # sometimes np.transpose is not really working as expected... but it should be ok with this
    aug_samples2 = np.transpose(aug_samples2, (2, 0, 1))  # get n,m,p

    diff = aug_samples1-aug_samples2
    dist = np.matmul(diff**2, np.ones((n, p, 1)))
    dist = np.sqrt(dist[:, :, 0])
    return dist


def display_features(dic_features):
    i_names = 0
    for i_feat in range(len(dic_features["features"])):
        feat = dic_features["features"][i_feat]
        name = dic_features["names"][i_names:i_names+feat.shape[-1]]
        i_names += feat.shape[-1]
        print(name, feat.shape)


def display_feature_importances(dic_features, clf):
    for i, feat_name in enumerate(dic_features["names"]):
        print(f"{feat_name} importance:", clf.feature_importances_[i])


def plot_and_save_confusion_matrix(pred_y, true_y, dst_path):
    cf_matrix = confusion_matrix(pred_y, true_y)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues')
    plt.savefig(dst_path)
    plt.close()


def plot_and_save_history(history, dst_path):
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(history.history['sparse_categorical_accuracy'],
             label="sparse_categorical_accuracy")
    plt.plot(history.history['val_sparse_categorical_accuracy'],
             label="val_sparse_categorical_accuracy")
    plt.legend()
    plt.show()
    plt.savefig(dst_path)
    plt.close()


def save_experiment_in_excel(dst_path, clf, train_clf_f1_score, k_fold_number, val_scores, dic_features):
    def get_dic_experiment(clf, train_clf_f1_score, k_fold_number, val_scores, dic_features):
        dic_experiment = {
            "training_info": {"f1_score": train_clf_f1_score},
            "validation_info": {"k_fold_number": k_fold_number, "scores_mean": val_scores.mean(), "scores_std": val_scores.std(), "scores_max": np.max(val_scores), "scores_min": np.min(val_scores)},
            "feature_importance_info": {f"{feat_name} importance": clf.feature_importances_[i] for i, feat_name in enumerate(dic_features["names"])}
        }
        return dic_experiment

    dic_experiment = get_dic_experiment(
        clf, train_clf_f1_score, k_fold_number, val_scores, dic_features)

    new_df = {}
    for main_key, sub_dic in dic_experiment.items():
        if main_key == "training_info":
            appendix = "train_"
        elif main_key == "validation_info":
            appendix = "val_"
        else:
            appendix = ""

        for sub_key, sub_value in sub_dic.items():
            new_df[appendix+sub_key] = sub_value

    new_df = pd.DataFrame(new_df, index=[0])

    if not (os.path.exists(dst_path)):
        df = new_df
    else:
        df_read = pd.read_excel(dst_path)
        df = df_read.append(new_df, ignore_index=True)

    df.to_excel(dst_path, index=False)
    print("experiment saved at "+dst_path)
    return dic_experiment
