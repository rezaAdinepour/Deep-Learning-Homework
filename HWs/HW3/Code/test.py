# coding=utf8

"""
    Neural net. Self-Organizing Map (SOM) or Kohonen Layer for unsupervised machine learning
"""

import logging
import numpy as np
from scipy import sparse
from scipy.sparse import issparse, csr_matrix

from sklearn.cluster.KMeans import _squared_norms
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils import atleast2d_or_csr
from sklearn.utils.extmath import safe_sparse_dot

import timeit

__author__ = 'zavsnar'


class SOMCluster(object):
    """
    Implementation clustering with Self-Organizing Map (SOM).
    Some ideas taken from mvpa2.suite.SimpleSOMMapper.

    Examples of causes
    ------------------
        items: 1000 learning_rate=0.0005, niter=50:      {(1, 0): 524, (0, 1): 282, (1, 1): 156, (0, 0): 38}
                    learning_rate=0.0005, niter=100:     {(0, 0): 466, (1, 1): 371, (1, 0): 163}
               10.000 radius=1, learning_rate=0.00005, niter=100:
                               {8: 4211, 6: 1867, 0: 1550, 2: 1216, 4: 1049, 7: 51, 1: 37, 5: 13, 3: 6} time: 341.54
    """
    def __init__(self, features, kshape, niter, learning_rate=0.00005, radius=None, weight_layer=None):
        """
        Parameters
        ----------
        kshape : (int, int)
          Shape of the internal Kohonen layer. Currently, only 2D Kohonen layers are supported.
        niter : int
          Number of iteration during network training.
        learning_rate : float
          Initial learning rate, which will continuously decreased during network training.
        radius : float or None
          Initial radius of the Gaussian neighborhood kernel radius,
          which will continuously decreased during network training.
          If `None` (default) the radius is set equal to the longest edge of the Kohonen layer.
        """
        self.kshape = kshape

        self.radius = radius or max(self.kshape)
        self.learning_rate = learning_rate
        self.niter = niter

        # precompute whatever can be done scalar for decay of learning rate and radius across all iterations
        self.iter_scale = self.niter / np.log(self.radius)

        self.feature_list = features

        # init weight_layer
        weight_layer_size = tuple(kshape) + (len(features),)
        if weight_layer:
            if weight_layer.shape != weight_layer_size:
                raise Exception(u'Initial weight_layer has shape {}, but must has {}.'
                                .format(weight_layer.shape, weight_layer_size))

            self.weight_layer = np.asarray(weight_layer, dtype=np.float)
        else:
            self.weight_layer = np.random.random_integers(low=0, high=100, size=weight_layer_size).astype(np.float)

        # precompute distance kernel between elements in the Kohonen layer
        # that will remain constant throughout the training
        # (just compute one quadrant, as the distances are symmetric)
        self.dist_in_Kohonen_layer = np.fromfunction(lambda x, y: (x**2 + y**2)**0.5, self.kshape, dtype='float')
        self.count_samples_by_clusters = [0] * self.kshape[0]*self.kshape[1]

    def predict(self, X):
        x_squared_norms = _squared_norms(sparse.csr_matrix([X]))
        #TODO: may be neurons must get flat list not 2d-matrix (self.weight_layer)
        bmu = self._get_bmu(X, neurons=self.weight_layer, x_squared_norm=x_squared_norms)
        return bmu

    def fit(self, samples, predict=True):
        """
        Parameters
        ----------
            samples: array_like or sparse matrix, shape=(n_samples, n_features)
        """
        X_clusters = []
        unit_deltas = np.zeros(self.weight_layer.shape, dtype='float')

        x_squared_norms = _squared_norms(sparse.csr_matrix(samples))

        tm_start = timeit.default_timer()

        for it in xrange(1, self.niter + 1):
            neighbors_influence = self._compute_influence_neighbors(it)
            weight_layer_flat = self.weight_layer.reshape(self.kshape[0]*self.kshape[1], len(self.feature_list))
            scr_mtx = atleast2d_or_csr(sparse.csr_matrix(weight_layer_flat))
            self.K_squared_norms = _squared_norms(scr_mtx)
            # self.K_squared_norms = None
            for i, sample in enumerate(samples):
                # determine closest neuron
                bmu = self._get_bmu(sample, weight_layer_flat, x_squared_norms[i])
                if predict and it == self.niter:
                    _cluster_idx = self.kshape[1]*bmu[0] + bmu[1]
                    X_clusters.append(_cluster_idx)
                    self.count_samples_by_clusters[_cluster_idx] += 1

                kshape = self.kshape
                infl_map = np.zeros(shape=kshape)

                # upper left (inclusive bottom-center)
                infl_map[:bmu[0]+1, :bmu[1]+1] = neighbors_influence[bmu[0]::-1, bmu[1]::-1]
                # upper right (inclusive bottom)
                infl_map[:bmu[0]+1, bmu[1]+1:] = neighbors_influence[bmu[0]::-1, 1:kshape[1] - bmu[1]]
                # bottom left (inclusive center)
                infl_map[bmu[0]+1:, :bmu[1]+1] = neighbors_influence[1:kshape[0] - bmu[0], bmu[1]::-1]
                # bottom right
                infl_map[bmu[0]+1:, bmu[1]+1:] = neighbors_influence[1:kshape[0] - bmu[0], 1:kshape[1] - bmu[1]]

                if issparse(sample):
                    sample = sample.toarray()[0]

                diff = (sample - self.weight_layer)
                unit_deltas += infl_map[:,:,np.newaxis] * diff

            # update weight_layer
            self.weight_layer += unit_deltas

            # reset unit deltas
            unit_deltas.fill(0.)

        logging.info(u'Kohonen net partial fitted for {}'.format(timeit.default_timer()-tm_start))

        if predict:
            return X_clusters

    def partial_fit(self, X, predict=True):
        return self.fit(X, predict=predict)

    def _compute_influence_neighbors(self, iteration):
        # Compute the neighborhood kernel for some iteration.

        # compute radius decay for this iteration
        curr_max_radius = self.radius * np.exp(-1.0 * iteration / self.iter_scale)

        # same for learning rate
        curr_learning_rate = self.learning_rate * np.exp(-1.0 * iteration / self.iter_scale)

        # compute Gaussian influence kernel
        neighbors_influence = np.exp((-1.0 * self.dist_in_Kohonen_layer) / (2 * curr_max_radius * iteration))
        neighbors_influence *= curr_learning_rate

        # hard-limit kernel to max radius
        neighbors_influence[self.dist_in_Kohonen_layer > curr_max_radius] = 0.

        return neighbors_influence

    def _get_bmu(self, X, neurons, x_squared_norm):
        """Returns the ID of the best matching unit (neuron).

        'best' is determined as minimal squared Euclidean distance between
        any units weight vector and some given target `sample`

        Returns
        -------
        tuple: (row, column)
        """
        _distances = euclidean_distances_optimised(X=X,
                                                   X_norm_squared=x_squared_norm,
                                                   Y=neurons,
                                                   Y_norm_squared=self.K_squared_norms)
        idx = np.argmin(_distances)

        # idx = np.argmin(((self.weight_layer - sample) ** 2).sum(axis=2))
        # assumes 2D Kohonen layer
        return (np.divide(idx, self.kshape[1]).astype('int'), idx % self.kshape[1])


def euclidean_distances_optimised(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
    """
    Updated function from sklearn.metrics.euclidean_distances
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as Y, then you should just pre-compute the output and not even
    # call this function.
    X, Y = check_pairwise_arrays(X, Y)
    if X_norm_squared is None:
        if issparse(X):
            XX = X.multiply(X).sum(axis=1)
        else:
            XX = np.sum(X * X, axis=1)[:, np.newaxis]
    else:
        XX = atleast2d_or_csr(X_norm_squared, force_all_finite=False)
        if XX.shape != (1, X.shape[0]):
            raise ValueError(u"Incompatible dimensions for X and X_norm_squared")

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is None:
        if issparse(Y):
            # scipy.sparse matrices don't have element-wise scalar
            # exponentiation, and tocsr has a copy kwarg only on CSR matrices.
            YY = Y.copy() if isinstance(Y, csr_matrix) else Y.tocsr()
            YY.data **= 2
            YY = np.asarray(YY.sum(axis=1)).T
        else:
            YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    else:
        YY = atleast2d_or_csr(Y_norm_squared, force_all_finite=False)
        if YY.shape != (1, Y.shape[0]):
            raise ValueError(u"Incompatible dimensions for Y and Y_norm_squared")

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    # np.maximum(distances, 0, distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances)