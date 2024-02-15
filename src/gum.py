"""Functions for fitting gaussian-uniform mixtures (GUMs) to data."""

from sklearn.mixture import GaussianMixture
from scipy.stats import norm, uniform
import numpy as np


class GUM:
    """Gaussian-Uniform Mixture, with help from : http://tinyurl.com/5epes88s"""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.quant_samples = 10_000
        self.prob_samples = np.arange(1.0, self.quant_samples) / (self.quant_samples)

        self.mu = np.zeros(shape=self.n_components)
        self.sig = np.zeros(shape=self.n_components)
        self.tau = np.ones(shape=self.n_components) / n_components
        self.unif_dist = uniform()

        self.cdf = lambda x, **args: self.distribution_stat(x, "cdf", np.sum, **args)
        self.pdf = lambda x, **args: self.distribution_stat(x, "pdf", np.sum, **args)
        self.ppf = lambda x: np.interp(x, self.prob_samples, self.quantiles)

    @property
    def dists(self):
        """Generate list of `scipy` distribution objects from fitted parameters
        as a class property"""
        return [
            *[norm(loc=mu, scale=np.sqrt(sig)) for (mu, sig) in zip(self.mu, self.sig)],
            self.unif_dist,
        ]

    @property
    def quantiles(self):
        """Generate numerical quantiles for this mixture's distribution using
        `self.quant_samples` number of samples points"""
        gmm_samples = self.sample(self.quant_samples)
        return np.quantile(gmm_samples, q=self.prob_samples)

    def distribution_stat(self, input_data, function, summary_func, flatten=True):
        """Generic function to execute `scipy` distribution function sub-function
        from each component of mixture and optionally accumulate with sum"""
        stats = [
            (getattr(dist, function)(input_data) * weight).squeeze()
            for (dist, weight) in zip(self.dists, self.tau)
        ]
        summarised = summary_func(stats, axis=0)
        return summarised.flatten() if flatten else summarised

    def sample(self, count=100):
        """Sample `count` points from the GUM"""
        indexes = np.random.choice(
            self.n_components + 1, size=count, replace=True, p=self.tau
        )
        return np.concatenate(
            [dist.rvs(size=sum(indexes == ind)) for ind, dist in enumerate(self.dists)]
        )

    def weighted_pdf(self, input_data):
        """Generate the PDF of each input datapoint belonging to each mixture component"""
        return self.distribution_stat(input_data, "pdf", np.stack, flatten=False)

    def expectation_step(self, input_data):
        """Expectation step of the EM Algorithm; generated normalised PDFs"""
        weighted_pdf = self.weighted_pdf(input_data)
        return weighted_pdf / np.sum(weighted_pdf, axis=0)

    def maximisation_step(self, input_data, posterior):
        """Maximisation step of  the EM Algorithm; update distribution parameters
        along analytical gradients using values from normalised PDF"""
        norm_z_posterior = posterior[:-1]
        posterior_sum = np.sum(posterior, axis=1)
        norm_posterior_sum = posterior_sum[:-1]

        new_tau = posterior_sum / input_data.shape[0]
        new_mu = ((norm_z_posterior @ input_data).T / norm_posterior_sum).T

        data_centered = input_data - self.mu.T
        data_weighted = data_centered * norm_z_posterior.T
        new_sig = np.diag(data_weighted.T @ data_centered) / norm_posterior_sum
        new_sig = new_sig[:, None, None]
        new_sig += 1e-6  # Stop covariance singularities
        delta_sig = np.min(np.abs(new_sig - self.sig))

        return (new_mu, new_sig, new_tau), delta_sig

    def fit(self, input_data, from_gmm=None, tol=0.0005, max_iterations=100):
        """Fit the mixture paramters to a set of input_data, initialising from
        a GMM (which can be provided)."""
        # Initialise from GMM (can be provided)
        if from_gmm is None:
            from_gmm = GaussianMixture(n_components=self.n_components).fit(input_data)
        self.mu = from_gmm.means_.flatten()
        self.sig = np.sqrt(from_gmm.covariances_)
        self.tau = np.concatenate((from_gmm.weights_.flatten(), np.array([0.1])))
        self.tau /= np.sum(self.tau)
        self.unif_dist = uniform(
            loc=np.min(input_data), scale=np.max(input_data) - np.min(input_data)
        )

        # Iterate EM until converged
        iteration, delta_sig = 0, np.inf
        while not (delta_sig < tol) or (iteration > max_iterations):
            z_posterior = self.expectation_step(input_data)
            new_params, delta_sig = self.maximisation_step(input_data, z_posterior)
            self.mu, self.sig, self.tau = new_params
            iteration += 1
        return self

    def log_likelihood(self, input_data):
        """Calculate the log-likelihood of provided samples"""
        return np.sum(np.log(np.sum(self.weighted_pdf(input_data), axis=0)))

    def bic(self, input_data):
        """BIC = k*ln(n)-2*ln(Likelihood)
        k=num_params, n=num_samples"""
        # 3 params for each gaussian and 1 weight for normal
        k = (self.n_components * 3) + 1
        n, _ = input_data.shape
        return k * np.log(n) - 2 * self.log_likelihood(input_data)


def find_bimodal_intercept(mu1, mu2, sd1, sd2, between_means=True):
    """Calulate the points of intersection between the PDFs of two normal
    distributions, with the option to keep only the one between the means."""
    a = (-1 / (sd1**2)) + (1 / (sd2**2))
    b = 2 * ((mu1 / (sd1**2)) - (mu2 / (sd2**2)))
    c = ((mu2**2) / (sd2**2)) - ((mu1**2) / (sd1**2)) + np.log((sd2**2) / (sd1**2))
    roots = np.roots([a, b, c])
    if between_means:
        means = np.sort(np.array([mu1, mu2]))
        return roots[(means[0] < roots) & (roots < means[1])]
    return roots


def find_unimodal_tail_intercept(data, model):
    """Calculate the intercepts between a single gaussian distribution and a
    normal distribution, returning the intercept that boundaries the tail with
    the most data in it.s"""
    mu, sd, norm_w, unif_w = (
        model.mu[0][0],
        np.sqrt(model.sig[0][0][0]),
        model.tau[0],
        model.tau[-1] / (max(data) - min(data)),
    )
    root = np.sqrt(-2 * np.log((unif_w * sd * np.sqrt(2 * np.pi)) / norm_w))
    s1, s2 = np.array([mu - sd * root, mu + sd * root])
    return s1 if sum(data < s1) > sum(data > s2) else s2


def gum_classify(input_data):
    """Use the GUM model to classify a dataset as `uni` or `bi` modal, detect
    the presence of a tail, and the value of the most probable threshold in the
    data."""
    all_models = np.array(
        [GaussianMixture(n_components=n).fit(input_data) for n in [1, 2]]
        + [GUM(n_components=n).fit(input_data) for n in [1, 2]]
    )
    all_bics = np.array([model.bic(input_data) for model in all_models])
    best_fit = np.argmin(all_bics)

    best_model = all_models[best_fit]
    num_normal = (best_fit % 2) + 1
    has_tail = best_fit >= 2

    threshold = None
    if num_normal > 1:
        if isinstance(best_model, GUM):
            mu1, mu2, sd1, sd2, _, _, _ = (
                *best_model.mu.flatten(),
                *np.sqrt(best_model.sig.flatten()),
                *best_model.tau.flatten(),
            )
        elif isinstance(best_model, GaussianMixture):
            mu1, mu2, sd1, sd2, _, _ = (
                *best_model.means_.flatten(),
                *np.sqrt(best_model.covariances_.flatten()),
                *best_model.weights_.flatten(),
            )
        threshold = find_bimodal_intercept(mu1, mu2, sd1, sd2)
        # threshold = find_bimodal_intercept(mu1, mu2, sd1, sd2, w1, w2)
    elif num_normal == 1 and has_tail:
        threshold = find_unimodal_tail_intercept(input_data, best_model)
    return (num_normal, has_tail, all_models[best_fit], threshold)


# def example_function(val: int):
#     return val**2
