from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_and_save_confusion_matrix(pred_y, true_y, dst_path):
    cf_matrix = confusion_matrix(pred_y, true_y)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues')
    plt.savefig(dst_path)
    plt.close()
