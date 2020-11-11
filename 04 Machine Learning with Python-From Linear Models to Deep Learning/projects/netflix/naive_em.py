"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape

    post = np.zeros((n, K))
    logL = 0.0

    for i in range(n):
        x  = X[i,:]
        phi= np.zeros(K)
        for j in range(K):
            mu = mixture.mu[j,:]
            var= mixture.var[j]

            p = np.exp(-1/(2*var) * (x-mu) @ (x-mu)) / (2*np.pi*var)**(d/2)
            phi[j] = p

        gamma = phi * mixture.p
        post[i, :] = gamma / gamma.sum()
        logL += np.log(gamma.sum())

    return post, logL


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = post.T @ X / post.sum(axis=0).reshape((K, 1))

    var = np.zeros(K)
    for i in range(K):
        var[i] = (post[:, i] @ np.square(X - mu[i]).sum(axis=1)) / (d * post[:, i].sum())

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    logL_old = None
    logL_new = None

    while (logL_old is None or (logL_new - logL_old) > (np.abs(logL_new) * 1e-6)):
        logL_old = logL_new
        post, logL_new = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, logL_new
