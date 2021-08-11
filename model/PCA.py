import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
from sklearn.decomposition import PCA


def PCAdecomposition(review_data, sentiment_data, index_data, vectorized_method):
    print('PCAing...')
    pca = PCA(n_components=3)
    pca.fit(review_data)
    review_new = pca.transform(review_data)
    positive_review = []
    negative_review = []
    new_review = []
    for index, item in enumerate(review_new):
        temp = [i for i in item]
        temp.append(index_data[index])
        new_review.append(temp)
    df_pos = DataFrame(columns=('original_index', 'current_dataindex'))
    df_neg = DataFrame(columns=('original_index', 'current_dataindex'))
    counter_pos = 0
    counter_neg = 0
    for index, item in enumerate(new_review):
        if sentiment_data[index] == 1:
            positive_review.append(item[:3])
            df_pos = df_pos.append({'original_index': item[3], 'current_dataindex': counter_pos}, ignore_index=True)
            counter_pos += 1
        else:
            negative_review.append(item[:3])
            df_neg = df_neg.append({'original_index': item[3], 'current_dataindex': counter_neg}, ignore_index=True)
            counter_neg += 1
    df_neg.to_csv('../../dataset/' + vectorized_method + '_PCA_neg_index_map.csv')
    df_pos.to_csv('../../dataset/' + vectorized_method + '_PCA_pos_index_map.csv')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(np.array(positive_review)[:, 0], np.array(positive_review)[:, 1], np.array(positive_review)[:, 2],
                 label='positive')
    ax.scatter3D(np.array(negative_review)[:, 0], np.array(negative_review)[:, 1], np.array(negative_review)[:, 2],
                 label='negative')
    plt.legend()
    plt.savefig(vectorized_method + '_PCA_visualization.png')
    np.save('Output/points/PCA_pos_' + vectorized_method + '.npy', positive_review)
    np.save('Output/points/PCA_neg_' + vectorized_method + '.npy', negative_review)
    plt.show()
