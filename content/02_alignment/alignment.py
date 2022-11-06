import numpy as np
from fastdtw import dtw, fastdtw
from scipy.spatial import distance as sp_dist

from scipy.spatial.distance import cdist


def pairwise_distance_matrix(
    X: np.ndarray, 
    Y: np.ndarray, 
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute pairwise distance matrix of two sequences
    
    Parameters
    ----------
    X : np.ndarray
        A 2D array with size (n_observations, n_features)
    Y : np.ndarray
        A 2D array with size (m_observations, n_features)
    metric: str
        A string defining a metric (see possibilities 
        in scipy.spatial.distance.cdist)

    Returns
    -------
    C : np.ndarray
        Pairwise cost matrix
    """
    if X.ndim == 1:
        X, Y = np.atleast_2d(X, Y)
        X = X.T
        Y = Y.T
    C = cdist(X, Y, metric=metric)
    return C

def accumulated_cost_matrix(C: np.ndarray) -> np.ndarray:
    """
    Dynamic time warping cost matrix from a pairwise distance matrix

    Parameters
    ----------
    D : double array
        Pairwise distance matrix (computed e.g., with `cdist`).

    Returns
    -------
    D : np.ndarray
        Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D


def optimal_warping_path(D):
    """
    Compute the warping path given an accumulated cost matrix

    Parameters
    ----------
    D: np.ndarray
        Accumulated cost Matrix
    
    Returns
    -------
    P: np.ndarray
        Optimal warping path
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)

def naive_dynamic_time_warping(
    X: np.ndarray, 
    Y: np.ndarray, 
    metric: str ='euclidean',
    return_distance: bool = False,
) -> (np.ndarray, float):
    """
    Naive Implementation of Vanilla Dynamic Time Warping
    
    Parameters
    ----------
    X : np.ndarray
        Array X
    Y: np.ndarray
        Array Y
    metric: string
        Scipy Metric
        
    Returns
    -------
    warping_path: np.ndarray
        The warping path for the optimal alignment.
    dtwd : float
        The dynamic time warping distance of the alignment.
    """
    # Compute pairwise distance matrix
    C = pairwise_distance_matrix(X, Y, metric=metric)
    # Compute accumulated cost matrix
    D = accumulated_cost_matrix(C)
    dtwd = D[-1, -1]
    # Get warping path
    warping_path = optimal_warping_path(D)
    
    if return_distance:
        return warping_path, dtwd
    return warping_path

def dynamic_time_warping(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    return_distance: bool = False,
) -> (np.ndarray, float):
    """
    Vanilla Dynamic Time Warping

    This is a more efficient implementation

    Parameters
    ----------
    X : np.ndarray
    Y: np.ndarray
    metric : str
       The name of the metric to use

    Returns
    -------
    warping_path: np.ndarray
        The warping path for the best alignment
    dtwd : float
        The dynamic time warping distance of the alignment.
    """
    # Get distance measure from scipy dist
    dist = getattr(sp_dist, metric)
    dtwd, warping_path = dtw(X, Y, dist=dist)

    # Make warping path a numpy array
    warping_path = np.array(warping_path)

    if return_distance:
        return warping_path, dtwd
    return warping_path, dtwd


def fast_dynamic_time_warping(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    return_distance: bool = False,
) -> (np.ndarray, float):
    """
     Fast Dynamic Time Warping

    This is an approximate solution to dynamic time warping.

    Parameters
    ----------
    X : np.ndarray
    Y: np.ndarray
    metric : str
       The name of the metric to use

    Returns
    -------
    warping_path: np.ndarray
        The warping path for the best alignment
    dtwd : float
        The dynamic time warping distance of the alignment.
    """

    # Get distance measure from scipy dist
    dist = getattr(sp_dist, metric)
    dtwd, warping_path = fastdtw(X, Y, dist=dist)

    # Make path a numpy array
    warping_path = np.array(warping_path)

    if return_distance:
        return warping_path, dtwd

    return warping_path, dtwd
