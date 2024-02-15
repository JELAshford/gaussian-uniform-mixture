"""Generators for synthetic distributions to test GUM classification rates"""

from scipy.stats import norm, uniform
import numpy as np


class Mixture:
    """`Mixture` of distributions that can be sampled from"""

    def __init__(self, dists=None, weights=None):
        self.dists = dists
        self.weights = weights

    def sample(self, count: int = 100):
        """Sample `count` points from the GUM"""
        weight_samp = np.random.choice(
            len(self.dists), count, replace=True, p=self.weights
        )
        return np.concatenate(
            [dist.rvs(sum(weight_samp == ind)) for ind, dist in enumerate(self.dists)]
        )


def random_unimodal(
    tail=None, num_sds=1, mean_range=(5, 10), sd_range=(0.5, 5), **kwargs
):
    num_normal = 1
    num_sds = num_sds if num_sds > 0 else np.random.randint(1, 7)

    norm_mean = np.random.uniform(*mean_range)
    norm_sd = np.random.uniform(*sd_range)
    dists = [norm(norm_mean, norm_sd)]
    weights = [0.95]
    if tail is not None:
        tail_weight = np.random.uniform(0.05, 0.25)
        weights = [1 - tail_weight, tail_weight]
        if tail == "left":
            width = num_sds * norm_sd
            edge = norm_mean - (2 * norm_sd) - width
        elif tail == "right":
            width = num_sds * norm_sd
            edge = norm_mean + (2 * norm_sd)
        else:
            raise ValueError(
                f"Tail type '{tail}' not recognised. Should be 'left' or 'right'."
            )
        dists.append(uniform(edge, width))

    weights = np.array(weights)
    weights /= np.sum(weights)
    return dists, weights, num_normal


def random_bimodal(
    tail=None,
    num_sds=1,
    mean_range=(5, 10),
    sd_range=(0.5, 5),
    minimum_spacing=False,
    **kwargs,
):
    num_normal = 2
    num_sds = num_sds if num_sds > 0 else np.random.randint(1, 7)

    low_norm_sd, high_norm_sd = np.random.uniform(*sd_range, size=2)
    if minimum_spacing:
        low_norm_mean = np.random.uniform(*mean_range, size=1)
        sd_sum = low_norm_sd + high_norm_sd
        high_norm_mean = low_norm_mean + np.random.uniform(
            sd_sum, sd_sum + mean_range[0], size=(1)
        )
    else:
        low_norm_mean, high_norm_mean = np.sort(np.random.uniform(*mean_range, size=2))

    dists = [norm(low_norm_mean, low_norm_sd), norm(high_norm_mean, high_norm_sd)]
    weight_offset = np.random.rand() / 10
    weights = np.array([0.5 + weight_offset, 0.5 - weight_offset])
    if tail is not None:
        tail_weight = np.random.uniform(0.05, 0.25)
        norm_weight = (1 - tail_weight) / 2
        weights = np.array([norm_weight, norm_weight, tail_weight])

        if tail == "left":
            width = num_sds * low_norm_sd
            edge = low_norm_mean - (2 * low_norm_sd) - width
        elif tail == "right":
            width = num_sds * high_norm_sd
            edge = high_norm_mean + (2 * high_norm_sd)
        else:
            raise ValueError(
                f"Tail type '{tail}' not recognised. Should be 'left' or 'right'."
            )
        dists.append(uniform(edge, width))
    return dists, weights, num_normal


def generate_samples(test_types, num_tests, num_samples):
    samples, labels, test_names = [], [], []
    for _ in range(num_tests):
        test_name = np.random.choice([*test_types.keys()], size=1)[0]
        dists, weights, num_normal = test_types[test_name]()
        test_label = [num_normal, "tail" in test_name]
        sample_count = num_samples if num_samples > 0 else np.random.randint(50, 150)
        samples.append(Mixture(dists, weights).sample(sample_count).reshape(-1, 1))
        labels.append(test_label)
        test_names.append(test_name)
    return samples, labels, test_names


def sim1_samples(
    num_tests=150,
    num_samples=-1,
    num_tail_sds=1,
    mean_range=(5, 10),
    sd_range=(0.5, 5),
):
    test_params = {
        "num_sds": num_tail_sds,
        "mean_range": mean_range,
        "sd_range": sd_range,
    }
    test_types = {
        "unimodal_right_tail": lambda: random_unimodal(tail="right", **test_params),
        "bimodal_right_tail": lambda: random_bimodal(tail="right", **test_params),
    }
    samples, labels, _ = generate_samples(test_types, num_tests, num_samples)
    return samples, labels


def sim2_samples(
    num_tests=150,
    num_samples=-1,
    num_tail_sds=-1,
    mean_range=(3, 10),
    sd_range=(0.5, 5),
):
    test_params = dict(num_sds=num_tail_sds, mean_range=mean_range, sd_range=sd_range)
    test_types = {
        "unimodal": lambda: random_unimodal(tail=None, **test_params),
        "unimodal_right_tail": lambda: random_unimodal(tail="right", **test_params),
        "bimodal": lambda: random_bimodal(tail=None, **test_params),
        "bimodal_right_tail": lambda: random_bimodal(tail="right", **test_params),
    }
    samples, labels, _ = generate_samples(test_types, num_tests, num_samples)
    return samples, labels


# Alternative methods that strays from the original simulations:
#
# Define a popuation of solutions by a set of random numbers between 0 and 1
# [p_is_bimodal, frac_m1, frac_m2, frac_s1, frac_s1, p_has_tail, p_left_right, frac_sd_range]
# ...
# if p_is_bimodal < prop_bimodal <unimodal> else <bimodal>
# if <unimodal>: m1 = mean_range * frac_m1; s1 = sd_range * frac_s1
# if <bimodal>: s1, s2 = sd_range * {frac_s1, frac_s2}; m1 = (mean_range - (s1+s2)) * frac_m1; m2 = frac_m2 * (mean_range - m1)
# <dists> = [normal(m, s), ..] ordered by m (loc)
# if p_has_tail < prop_has_tail <has_tail> else <no_tail>
# if <has_tail>: <tail_side> = <left> if p_left_right < p_left_frace else <right>
# <side_sd> = <dists>[0].scale if <left> else <dists>[1].scale
# <tail_len> = <side_sd> * frac_sd_range * sd_range
# <side_loc> = <dists>[0].loc-2*<side_sd> - tail_len if <left> else <dists>[1].scale+ 2*<side_sd>
# <dists>.append(uniform(loc=<side_loc>, scale=<tail_len>))
# store dists for later comparison and sample points from dists
