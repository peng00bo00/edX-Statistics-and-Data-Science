"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    def logGaussian(x, mu, var):
        """
        A helper function to get log probability of multivariate normal distribution.

        Args:
            x: the random variable
            mu: the mean vector
            var: the variance
        
        Returns:
            log_prob: the log probability
        """

        d = len(x)
        log_prob = -1/(2*var) * (x-mu) @ (x-mu)
        log_prob-= d/2 * np.log(2*np.pi*var)

        return log_prob

    n, d = X.shape
    K, _ = mixture.mu.shape

    log_post = np.zeros((n, K))
    logL = 0.0

    for i in range(n):
        mask = (X[i,:] != 0)
        x = X[i, mask]
        for j in range(K):
            mu = mixture.mu[j, mask]
            var= mixture.var[j]

            log_prob = logGaussian(x, mu, var)
            log_post[i, j] = np.log(mixture.p[j] + 1e-16) + log_prob
        
        log_sum = logsumexp(log_post[i, :])
        log_post[i, :] -= log_sum
        logL += log_sum

    return np.exp(log_post), logL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = mixture.mu.copy()
    var= np.zeros(K)

    for j in range(K):
        var_num = 0
        var_den = 0

        for l in range(d):
            mask = (X[:, l] != 0)
            count = post[mask, j].sum()
            ## update mu
            if count >= 1:
                mu[j, l] = (X[mask, l] @ post[mask, j]) / count

            var_num += np.square(mu[j, l] - X[mask, l]) @ post[mask, j]
            var_den += count

        ## update var
        var[j] = var_num / var_den
        if var[j] < min_variance:
            var[j] = min_variance

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
        mixture = mstep(X, post, mixture)

    return mixture, post, logL_new


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """

    def logGaussian(x, mu, var):
        """
        A helper function to get log probability of multivariate normal distribution.

        Args:
            x: the random variable
            mu: the mean vector
            var: the variance
        
        Returns:
            log_prob: the log probability
        """
        d = len(x)
        log_prob = -1/(2*var) * (x-mu) @ (x-mu)
        log_prob-= d/2 * np.log(2*np.pi*var)

        return log_prob

    n, d = X.shape
    K, _ = mixture.mu.shape
    X_pred = X.copy()

    for i in range(n):
        mask = (X[i, :] != 0)

        post = np.zeros(K)
        for j in range(K):
            log_prob = logGaussian(X[i, mask], mixture.mu[j, mask], mixture.var[j])
            post[j]  = np.log(mixture.p[j] + 1e-16) + log_prob

        post = np.exp(post - logsumexp(post))
        X_pred[i, ~mask] = post @ mixture.mu[:, ~mask]

    return X_pred
