import numpy as np
import sklearn.metrics as metrics
import distance as d


def find_nearest_cluster(dataset, labels, sample_label, sample):
    r"""Find the nearest cluster available from a given sample, outside its own cluster.
    This function uses a distance function to evaluate the distance from 
    a given sample and all other clusters in the dataset.
    The label of the cluster minimising the distance with the given sample is then returned
    
    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset
        sample_label (int): label of the sample for which we have to find the closest cluster
        sample (list of float): sample features

    Returns:
        label (int): ID of the closest cluster
    """
    unique_labels, labelled_samples = get_labelled_data(dataset, labels)

    distances = []
    for idx in range(len(unique_labels)):
        if unique_labels[idx] == sample_label:
            distances.append(-1)
        else:
            dist = 0
            for s in labelled_samples[idx]:
                dist += d.euclidean(s, sample)
            distances.append(dist/len(labelled_samples[idx]))
    
    min_dist = np.min([dist for dist in distances if dist > 0])
    closest_cluster = unique_labels[distances.index(min_dist)]

    return closest_cluster

def get_labelled_data(dataset, labels):
    r"""Arrange a dataset depending on the available clusters.
    It groups all sample issued from a similar cluster together.
    By doing so, it is possible to work on each cluster contained in a dataset.

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        unique_labels (list of int): Ordered list of labels available in the dataset
        labelled_sample (list of list of [float]): List containing the dataset clusters discretised. 
                                                   Each list corresponds to a unique cluster.
                                                   The cluster list contains all sample related to this cluster.
    """
    unique_labels = np.unique(labels)
    labelled_samples = []

    for label in unique_labels:
        labelled_samples.append([s for s,l in zip(dataset,labels) if l==label])
    return unique_labels, labelled_samples

def get_centroid(cluster):
    r"""Get the centroid point of a cluster.
    The centroid is taken by summing each feature of all samples in the cluster, and then taking the average.

    Args:
        cluster (list of [float]): Samples contained in the cluster.

    Returns:
        centroid [list of float]: Centroid coordinates for each dimension.
    """
    sample_count = len(cluster)
    centroid = [sum(f)/sample_count for f in zip(*cluster)]
    return centroid

def silhouette_score(dataset, labels):
    r"""Compute the silhouette score for a clustered dataset.
    The silhouette score for a unique sample measures how well this sample is assigned to its cluster by comparing it with other clusters.
    Its range from [-1, 1], with a higher score indicating that a sample is well assigned to its cluster.
    The silhouette score of an entire dataset is the average of the silhouette score for all samples. 

    It is defined as follow:
        First, we evaluate how well the sample i is assigned to its cluster C_I:
            score_{intra}(i) = \frac{1}{|C_I| - 1} \sum_{j \in C_I, i \neq j} dist(i,j)
        
        Then, we evaluate the dissimilarities between the sample i and an other cluster C_J.
        The goal is to find the least dissimilar cluster from the sample i (aka, the "neighboring cluster"), hence:
            score_{inter}(i) = \min_{J \neq I} \frac{1}{|C_J|} \sum_{j \in C_J} dist(i,j)
        
        Finally, the silhouette score for a unique sample is defined as:
            silhouette(i) = \frac{score_{inter}(i) - score_{intra}(i)}{\max\{score_{intra}(i), score_{inter}(i)\}}
            
        The silhouette score of a whole clustere dataset can be defined as:
            silhouette = \frac{1}{N} \sum_{i=1}^{N} silhouette(i)

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        silhouette score (int): Silhouette score for the entire clustered dataset.
    """
    sample_score = []
    for i in range(len(dataset)):
        sample_label = labels[i]
        current_sample = dataset[i]
        nearestcluster = find_nearest_cluster(dataset, labels, sample_label, current_sample)
        
        intracluster_score, nearestcluster_score = 0, 0
        
        # intra_score = (1 / |Ci|-1) * sum( dist(i,j) ) with Ci cluster of i
        idx = 0
        for sample in [s for s,l in zip(dataset,labels) if l==sample_label]:
            if(sample is not current_sample):
                intracluster_score += d.euclidean(current_sample, sample)
            idx += 1
        intracluster_score /= (idx-1)
        
        # nearest_score = (1 / |Cj|) * sum( dist(i,j) ) with Cj nearest cluster from i
        idx = 0
        for sample in [s for s,l in zip(dataset,labels) if l==nearestcluster]:
            nearestcluster_score += d.euclidean(current_sample, sample)
            idx += 1
        nearestcluster_score /= idx
        
        score = (nearestcluster_score - intracluster_score)/(max(intracluster_score, nearestcluster_score))
        sample_score.append(score)

    return np.mean(sample_score)

def BCSS(dataset, labels):
    r"""Compute the Between Cluster Sum of Squares (BCSS) for a clustered dataset.
    It can be described as the separation score between all clusters in a dataset.
    It is the weighted sum of squared distances between each cluster centroid, with the overall data centroid.
    It ranges from [0, +inf[, with a higher score indicating that the clusters are well separated from each other.

    It is defined as follow:
         BCSS = \sum_{i=1}^{k} n_i ||c_i - c||^2

         with n_i the number of  points in the cluster C_i, 
            c_i the centroid of C_i and c the centroid of all samples in the dataset.

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        BCSS score (float): BCSS score of the clustered dataset.
    """
    _, labelled_samples = get_labelled_data(dataset, labels)
    data_centroid = get_centroid(dataset)

    cluster_score = []
    for cluster in labelled_samples:
        sample_count = len(cluster)
        cluster_centroid = get_centroid(cluster)
        cluster_score.append(sample_count * (d.euclidean(data_centroid, cluster_centroid)**2))
    return np.sum(cluster_score)

def WCSS(dataset, labels):
    r"""Compute the Within Cluster Sum of Squares (WCSS) for a clustered dataset.
    It can be described as the cohesion score between all clusters in a dataset.
    It is the sum of squared distance between the samples and their respective cluster centroids.
    It ranges from [0, +inf[, with a smaller score indicating more compact and cohesive clusters.

    it is defined as follow:
        WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - c_i||^2

         with c_i the centroid of C_i

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        _type_: _description_
    """
    _, labelled_samples = get_labelled_data(dataset, labels)

    cluster_score = []    
    for cluster in labelled_samples:
        score = 0
        cluster_centroid = get_centroid(cluster)
        for sample in cluster:
            score += d.euclidean(cluster_centroid, sample)**2
        cluster_score.append(score)
    
    return np.sum(cluster_score)

def calinski_harabasz_score(dataset, labels):
    r"""Compute the Calinski-Harabasz index for a clustered dataset.
    It is also known as the Variance Ratio Criterion (VRC).
    This score measures how similar a sample is to its own cluster (cohesion), compared to other clusters (separation).
    It does so by calculating the weighted 0ratio between the separation and the cohesion of each samples in the dataset.
    It ranges from [0, +inf[, with a higher Calinski-Harabasz index indicating that the clusters are dense and well separated.

    It is defined as follow:
        First calculate the weighted separation and cohesion scores. 
        It corresponds the Between Cluster Sum of Squares BCSS (separation) and Withing Cluster Sum of Squares WCSS (cohesion).

        The weighted separation score corresponds to the BCSS score normalised by its degrees of freedom:
            separation = \frac{BCSS}{k-1} 
        
        The weighted cohesion score corresponds to the WCSS score normalised by its degrees of freedom:
            cohesion = \frac{WCSS}{n-k}
            
        Finally, the Calinski-Harabasz score is defined as:
            CH = \frac{separation}{cohesion}
    
    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        Calinski-Harabasz (int): Calinski-Harabasz score of the clustered dataset.
    """
    sample_count = len(dataset)
    cluster_count = len(np.unique(labels))
    BCSS_score = BCSS(dataset,labels)
    WCSS_score = WCSS(dataset,labels)

    separation = BCSS_score/(cluster_count - 1)
    cohesion = WCSS_score/(sample_count - cluster_count)

    return separation / cohesion

def get_cluster_avg_distance(cluster):
    r"""Compute the average distance between all samples in a cluster and its centroid.
    It is defined as:
        d_i = \frac{1}{N} \sum_{j=1}^{N} dist(c_i, x^i_j)

    Args:
        cluster (list of [float]): All samples contained in the cluster.

    Returns:
        distance (float): Average distance inside the cluster.
    """
    distance = 0
    centroid = get_centroid(cluster)
    for sample in cluster:
        distance += d.euclidean(sample, centroid)
    distance /= len(cluster)  
    
    return distance

def cluster_similarity(dataset, labels):
    r"""Compute the similarity score between each cluster of the dataset.
    Each cluster is assigned a similarity score with all the other clusters.
    Here, the similarity score between two clusters is defined as a trade of 
    between the average distance in each cluster and the distance between the two clusters.
    It ranges from [0, +inf[, with a lower score indicating a better seperation and tightness of the clusters.

    It is defined as:
        S_{ij} = \frac{d_i + d_j}{dist(c_i, c_j)}

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        similarities (list of [float]): Return the similarities between each cluster.
                                        The list represents each cluster.
                                        Then each list represent the similarity scores for this cluster in regards
                                        to all the other clusters.
    """
    _, labelled_samples = get_labelled_data(dataset, labels)

    similarities = []
    for idx in range(len(labelled_samples)):
        current_cluster = labelled_samples[idx]
        current_centroid = get_centroid(current_cluster)

        # Calculate cluster average distance -> The average distance between each samples in the cluster with its centroid
        intra_cluster_avg_distance = get_cluster_avg_distance(current_cluster)
        
        other_clusters = [clust for i,clust in enumerate(labelled_samples) if i!=idx]
        cluster_similarity = []
        for clust in other_clusters:
            # Calculate distance between the the two clusters
            other_cluster_centroid = get_centroid(clust) 
            centroids_distance = d.euclidean(current_centroid, other_cluster_centroid) 
            
            # Calculate the second cluster average distance
            other_cluster_avg_distance = get_cluster_avg_distance(clust)

            # Calculate the similarity score between two clusters
            score = (intra_cluster_avg_distance + other_cluster_avg_distance) / centroids_distance
            cluster_similarity.append(score)
        similarities.append(cluster_similarity)

    return similarities

def davies_bouldin_index(dataset, labels):
    r"""Compute the Davies-Bouldin index for the clustered dataset.
    This score measures the seperation between the difference clusters.
    It is done by averaging the maximimum of the similarity score between each clusters. 
    As a high similarity score indicates less well seperated clusters, by taking the maximum,
    we evaluate the worst case possible in terms of cluster seperation.
    It ranges between [0, +inf[, with a lower score indicating a better seperation between clusters.

    It is defined as:
        DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i}(S_{ij})

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        DB score (float): Davies-Bouldin score for the clustered dataset.
    """
    similarities = cluster_similarity(dataset, labels)
    cluster_count = len(np.unique(labels))
    max_similarities = [max(sim) for sim in similarities]

    return (np.sum(max_similarities)) / cluster_count
    