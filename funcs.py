import numpy as np
# from scipy.spatial.distance import cdist
from numba import njit, prange


@njit(cache=True)
def simulate(boids, D, perception, asp, coeffs):
    """
    Calculates data for the next step of process
    :param boids: input array which contains boids' data;
           D: distances matrix;
           perception: parameter of the 'visible' zone of boid;
           asp: height and width ratio;
           coeffs: parameters of the simulation functions
    :return: none
    """
    calc_dist(boids[:, :2], D)
    M = D < perception
    np.fill_diagonal(M, False)
    wa = wall_avoidance(boids, asp)
    nz = noise(boids)
    for i in prange(boids.shape[0]):
        idx = np.where(M[i])[0]
        accels = np.zeros((5, 2))
        if idx.size > 0:
            accels[0] = alignment(boids, i, idx)
            accels[1] = cohesion(boids, i, idx)
            accels[2] = separation(boids, i, idx, D)
        accels[3] = wa[i]
        accels[4] = nz[i]
        # clip_mag(accels, *arange)
        boids[i, 4:6] = np.sum(accels * coeffs.reshape(-1, 1), axis=0)


@njit(cache=True)
def mean_axis0(arr):
    """
    Return mean for each column of 2D array
    :param arr: input array, at least 2D
    :return: res: output array 1D which contains mean values of all rows of arr
    """
    n = arr.shape[1]
    res = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        res[i] = arr[:, i].mean()
    return res


@njit
def clip_mag(arr, low, high):
    """
    Clips vectors in range (low, high) (will use for velocities)
    :param: arr: data array;
            low: bottom border of vectors value;
            high: top border of vectors value
    :return: none
    """
    mag = np.sum(arr * arr, axis=1) ** 0.5
    mask = mag > 1e-16
    mag_cl = np.clip(mag[mask], low, high)
    arr[mask] *= (mag_cl / mag[mask]).reshape(-1, 1)


@njit(cache=True)
def init_boids(boids, asp, vrange=(0., 1.), seed=0):
    """
    Initialise coordinates and velocities vectors of boids
    :param: boids: input array which contains boids' data;
            asp: height and width ratio;
            vrange: range of initial boids' velocities
            seed: random generation seed
    :return: none
    """

    N = boids.shape[0]
    np.random.seed(seed)
    boids[:, 0] = np.random.rand(N) * asp
    boids[:, 1] = np.random.rand(N)
    v = np.random.rand(N) * (vrange[1] - vrange[0]) + vrange[0]
    alpha = np.random.rand(N) * 2 * np.pi
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


@njit(cache=True)
def directions(boids):
    """
    Calculates vectors of directions of boids
    :param boids: input array which contains boids' data
    :return: np.hstack((boids[:, :2] - boids[:, 2:4], boids[:, :2])):
             output 2D-array which contains coordinates of directions vectors
    """
    return np.hstack((boids[:, :2] - boids[:, 2:4], boids[:, :2]))


@njit(cache=True)
def propagate(boids, dt, vrange):
    """
    Calculates coordinates and velocities for the next step of process
    :param:boids: input array which contains boids' data;
           dt: step time;
           vrange: range of initial boids' velocities
    :return: none
    """
    boids[:, :2] += dt * boids[:, 2:4] + 0.5 * dt ** 2 * boids[:, 4:6]
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], vrange[0], vrange[1])


@njit(cache=True)
def calc_dist(arr, D):
    """
    Calculates distances between dots of an input array
    :param: arr: input array which contains dots coordinates;
           D: matrix of distances we want to calc
    :return: none
    """
    n = arr.shape[0]
    for i in prange(n):
        for j in range(i):
            v = arr[j] - arr[i]
            d = (v @ v) ** 0.5
            D[i, j] = d
            D[j, i] = d


@njit(cache=True)
def periodic_walls(boids, asp):
    """
    Fills boids inside the box
    :param: boids: input array which contains boids' data;
            asp: height and width ratio
    :return: none
    """
    boids[:, :2] %= np.array([asp, 1.])


@njit(cache=True)
def alignment(boids, i, idx):
    """
    Aligns velocity of boid
    :param: boids: input array which contains boids' data;
            i: current boid index;
            idx: neighbours' indices
    :return: a: array of difference between mean velocity of neighbours and current boid
    """
    avg = mean_axis0(boids[idx, 2:4])
    a = avg - boids[i, 2:4]
    return a


@njit(cache=True)
def cohesion(boids, i, idx):
    """
    Makes boids closer to each other
    :param: boids: input array which contains boids' data;
            i: current boid index;
            idx: neighbours' indices
    :return: a: array of difference between mean coordinate of neighbours and current boid
    """
    center = mean_axis0(boids[idx, 0:2])
    a = center - boids[i, 0:2]
    return a


@njit(cache=True)
def separation(boids, i, idx, D):
    """
    Separates boids
    :param: boids: input array which contains boids' data;
            i: current boid index;
            idx: neighbours' indices;
            D: distances matrix
    :return: a: array which is powered sum of relative differences between distances of neighbours and current boid
    """
    d = boids[i, 0:2] - boids[idx, 0:2]
    a = np.sum(d / (D[i][idx].reshape(-1, 1)) ** 0.75, axis=0)
    return a


@njit(cache=True)
def wall_avoidance(boids, asp):
    """
    Keeps boids inside the box
    :param: boids: input array which contains boids' data;
            asp: height and width ratio
    :return: none
    """
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])

    ax = 1 / left ** 2 - 1 / right ** 2
    ay = 1 / bottom ** 2 - 1 / top ** 2

    return np.column_stack((ax, ay))


@njit(cache=True)
def noise(boids):
    """
    Generates 'noise' what refer to unaccounted factors
    :param:boids: input array which contains boids' data
    :return: np.random.rand(boids.shape[0], 2) * 2 - 1:
             1D-array which contains noise data
    """
    return np.random.rand(boids.shape[0], 2) * 2 - 1