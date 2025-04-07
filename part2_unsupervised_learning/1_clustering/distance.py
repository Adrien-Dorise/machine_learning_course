import math


def euclidean(sample1, sample2):
    """Caculate the euclidean distance between two data point.
    The formula is:
        dist(X,Y) = \sqrt{\sum_{i=1}^{n}{ ( x_i - y_i ) ^ 2}}

    Args:
        sample1 (list of float): Features assigned to the first sample
        sample2 (list of float): Features assigned to the second sample

    Returns:
        distance (float): Euclidean distance between the two samples
    """
    sum = 0
    for idx in range(len(sample1)):
        sum += (sample1[idx] - sample2[idx])**2
    return math.sqrt(sum)