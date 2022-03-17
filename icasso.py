# -*- coding: utf-8 -*-

"""ICASSO signal decomposition"""

# Authors: Anna Padée <anna.padee@unifr.ch>
#
# License: BSD-3-Clause


import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.signal import find_peaks


from sklearn.decomposition import FastICA
from sklearn.metrics import pairwise_distances
from sklearn.utils import resample
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.spatial.distance as ssd

warnings.filterwarnings('error', category=ConvergenceWarning)

log = logging.getLogger('icasso')
logging.basicConfig(level=logging.INFO)


class icasso:
    def __init__(self, bootstrapping=True, random_seeds=True, contrast_function='logcosh', n_iterations=10):
        #parameters
        self.bootstrapping = bootstrapping
        self._random_seeds = random_seeds
        self._n_iterations = n_iterations
        self._iterations_completed = 0
        self.n_components = 0
        self.contrast_function = contrast_function
        self.FastICA_iter = 1000
        self.tol = 0.0001

        #results
        self._corr_matrix = np.empty((0, 0))
        self._distance_matrix = np.empty((0, 0))
        self.Wmatrix = np.empty((0, 0))
        self.Smatrix = np.empty((0, 0))
        self._clusters = []
        self._cluster_scores = []
        self._centrotypes = np.empty((0, 0))
        self.icasso=None

    def bootstrap(self, X):
        """
        Bootstrapping function. Randomly resamples the time series along the 0
        axis. All time series are resampled the same way.
        """
        indices = list(range(0, X.shape[0]))
        indices = resample(indices, replace=True)
        X_ = X[indices, :]
        return X_

    def decomposition(self, X, n_components, seed=None):
        """
        Performs FastICA decomposition into n components.Returns demixing
        matrix W and sources S
        """
        ica = FastICA(n_components=n_components, random_state=seed, fun=self.contrast_function, max_iter=self.FastICA_iter, tol=self.tol)
        ica.fit_transform(X)  # Get the estimated sources
#        A_ = ica.mixing_
#        W_ = np.linalg.pinv(A_)
        W_ = ica.components_
        return W_

    def fit(self, X, n_components=None):
        """
        Performs the decomposition n_iterations times. If random_seeds are
        selected, each FastICA iteration is initialized from current time.
        Otherwise, with consecutive numbers. If bootstraping was selected, data
        is bootstrapped before every iteration.
        Returns stacked demixing matirces W.
        """
        log.info("Signals shape: " + str(X.shape))
        if n_components is None:
            n_components = X.shape[1]
        self.n_components = n_components

        self.Wmatrix = np.empty((0, X.shape[1]))
        self.Smatrix = np.empty((X.shape[0], 0))

        for i in range(0, self._n_iterations):
            if self.bootstrapping:
                _X = self.bootstrap(X)
            else:
                _X = X
            if self._random_seeds:
                seed = int(time.time()*1000000) & 0x00ffffffff
            else:
                seed = i
            warnings.filterwarnings('error', category=ConvergenceWarning)
            try:
                Wr = self.decomposition(_X, n_components, seed)
                self.Wmatrix = np.vstack((self.Wmatrix, Wr))
                self.n_components = Wr.shape[0]
                Sr = np.dot(X, Wr.T)
                for j in range(0, Sr.shape[1]):
                    Sr[:, j] = (Sr[:, j] - np.mean(Sr[:, j])) / np.sqrt(np.var(Sr[:, j]))
                self.Smatrix = np.hstack((self.Smatrix, Sr))
                log.debug("ICA fit " + str(i+1) + " of " + str(self._n_iterations) + " completed.")
            except ConvergenceWarning:
                log.debug("ICA fit " + str(i + 1) + " of " + str(self._n_iterations) + " did not converge, skipping.")
            except ValueError:
                log.debug("ICA fit " + str(i + 1) + " of " + str(self._n_iterations) + " produced NaN results, skipping.")
        self._iterations_completed = int(self.Wmatrix.shape[0] / self.n_components)

        log.info("Completed  " + str(self._iterations_completed) + " iterations out of " + str(self._n_iterations))
        warnings.filterwarnings('default', category=ConvergenceWarning)
        return self.Wmatrix, self.Smatrix

    def compute_similarities(self, mode="correlation"):
        """
            Computes distance matrix between components.
        Args:
            mode: (str) "correlation" or distance metric for sklearn.metrics.pairwise_distances

        Returns: (np.array) Distance matrix
        """
        if mode == "correlation":
            self._corr_matrix = np.corrcoef(self.Wmatrix)
            self._distance_matrix = -1 * np.abs(self._corr_matrix) + 1
            log.info("Distance matrix done: " + str(self._distance_matrix.shape))
        else:
            self._corr_matrix = np.corrcoef(self.Wmatrix)
            self._distance_matrix = pairwise_distances(self.Wmatrix, metric=mode)
            log.info("Distance matrix done: " + str(self._distance_matrix.shape))
        return self._distance_matrix

    def clustering(self, num_of_clusters=None):
        """
        Compute dendrogram of components and split into custers
        """
        self._cluster_scores = []
        if num_of_clusters==None:
            num_of_clusters = int(self.Wmatrix.shape[0] / self._iterations_completed)
        Z = linkage(ssd.squareform(self._distance_matrix, checks=False), method='ward')
        clusters = fcluster(Z, t=num_of_clusters, criterion='maxclust')
        self._clusters = clusters

        similarity_matrix = np.ones(self._distance_matrix.shape) - self._distance_matrix
        self._centrotypes = np.empty((0, self.Wmatrix.shape[1]))
        #compute scores

        for i in range(1, num_of_clusters+1):
            indices_in, = np.where(clusters == i)
            indices_out, = np.where(clusters != i)
            similarity_in = 0
            similarity_out = 0
            for j in indices_in:
                similarity_in += np.sum(similarity_matrix[indices_in, j])
                similarity_out += np.sum(similarity_matrix[indices_out, j])
            score = similarity_in/(len(indices_in)*len(indices_in)) - similarity_out/(len(indices_in)*len(indices_out))
            self._cluster_scores.append(score)

            signs = np.sign(self._corr_matrix[indices_in[0], indices_in])
            cluster_components = self.Wmatrix[indices_in, :]
            for j in range(0, len(signs)):
                cluster_components[j, :] = cluster_components[j, :] * signs[j]
            centrotype = np.sum(cluster_components, axis=0) / len(indices_in)
            self._centrotypes = np.vstack((self._centrotypes, centrotype))
        log.info("Clustering finished, cluster scores and centrotypes acquired for " + str(num_of_clusters) + " clusters")
        return Z, clusters, self._cluster_scores

    def best_decomposition(self):
        iteration_scores = []
        best_score = 0
        best_ind = 0
        for i in range(0, self._iterations_completed):
            score = 0
            for j in range(0, self.n_components):
                score += self._cluster_scores[self._clusters[(i * self.n_components) + j] - 1]
            iteration_scores.append(score)
            if score > best_score:
                best_score = score
                best_ind = i
        log.info("Best decomposition found: " + str(best_ind))
        return best_ind, iteration_scores




