import torch
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


def group_cluster(x, group=32, cluster_method='k_means'):
    # x : (torch tensor with shape [1, c, h, w])
    #batch size (b), channels (c), height (h), and width (w).

    xs = x.detach().cpu() # Detaching the tensor and moving it to the CPU is necessary for processing the tensor with scikit-learn, which does not work on GPU tensors
    b, c, h, w = xs.shape
    xs = xs.reshape(b, c, -1).reshape(b*c, h*w) #This reshapes the tensor twice: first to collapse the height and width into a single dimension, and then to merge the batch and channels dimensions.
    #Reshaping prepares the tensor for clustering by flattening it into a 2D array where each row represents a feature vector.
    #The feature vector is the pixel values of a single channel in a single image in the batch.
    #The number of rows is equal to the number of channels times the number of images in the batch.
    #The number of columns is equal to the height times the width of the image.
    #The result is a 2D array with shape [b*c, h*w] that contains all the feature vectors for all the images in the batch.
    #The next step is to cluster the feature vectors into groups.
    #The number of groups is equal to the number of channels we want in the output.
    #The result of clustering is a label for each feature vector that indicates which group it belongs to.
    #The label is an integer between 0 and n_clusters-1, where n_clusters is the number of groups.
    #The label is used to group the feature vectors into groups.
    #The result is a 2D array with shape [n_clusters, h*w] that contains the feature vectors for each group.
  

    if cluster_method == 'k_means':
        n_cluster = KMeans(n_clusters=group, random_state=0).fit(xs)
    elif cluster_method == 'agglomerate':
        n_cluster = AgglomerativeClustering(n_clusters=group).fit(xs)
    else:
        assert NotImplementedError

    labels = n_cluster.labels_
    del xs
    return labels


def group_sum(x, n=32, cluster_method='k_means'):
    #batch size (b), channels (c), height (h), and width (w).

    b, c, h, w = x.shape
    group_idx = group_cluster(x, group=n, cluster_method=cluster_method)
    init_masks = [torch.zeros(1, 1, h, w).to(x.device) for _ in range(n)]#This initializes a list of zero tensors, one for each cluster, with the same height and width as the input tensor.
    #Prepares masks for accumulating sums within each cluster.
    for i in range(c):
        idx = group_idx[i]#This gets the cluster index for the current channel.
         #Identifies the cluster to which the current channel belongs.

        init_masks[idx] += x[:, i, :, :].unsqueeze(1)#This adds the current channel's data to the corresponding cluster mask.
#Aggregates the data within the cluster.
    return init_masks
