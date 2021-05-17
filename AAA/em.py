import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.stats as ss
from sklearn import mixture
import sklearn
from mixture import MixtureModel
from scipy.stats import norm
from tqdm import tqdm

np.random.seed(17)

mixtura_original = mixture.GaussianMixture(
    n_components=2,
    covariance_type="spherical",
    weights_init=np.array([0.4, 0.6]),
    means_init=np.array([[-6], [2]]),
    precisions_init=np.array([0.25, 0.25]),  # 1/4 inverse of covariance
    random_state=17,
)
mixtura_original.covariances_ = np.array([4, 4])
mixtura_original.means_ = np.array([[-6], [2]])
mixtura_original.weights_ = np.array([0.4, 0.6])
mixtura_original.precisions_ = np.array([0.25, 0.25])
mixtura_original.fit_ = True


estim = mixture.GaussianMixture(
    n_components=3,
    covariance_type="spherical",
    weights_init=np.array([0.7, 0.1, 0.2]),
    means_init=np.array([[-6], [2], [0]]),
    precisions_init=np.array([0.25, 0.25, 0.25]),  # 1/4 inverse of covariance
    random_state=17,
)


def experiment(samples, regul_param):

    estim.covariances_ = np.array([4, 4, 4])
    estim.means_ = np.array([[-6], [2], [0]])
    estim.weights_ = np.array([0.7, 0.1, 0.2])
    estim.precisions_cholesky_ = np.array([0.5, 0.5, 0.5])
    estim.fit_ = True

    avg_log_likelihood_prev = -10e9

    # 1000 iteraciones máximo
    for i in range(1, 1000):
        probs_act = estim.predict_proba(samples)
        numerador_1 = 0
        numerador_2 = 0
        numerador_3 = 0
        for p in probs_act:  # sum over M, number of samples
            numerador_1 += p[0] * (1 + regul_param * p[0])
            numerador_2 += p[1] * (1 + regul_param * p[1])
            numerador_3 += p[2] * (1 + regul_param * p[2])
        pi_1 = numerador_1 / (numerador_1 + numerador_2 + numerador_3)

        pi_2 = numerador_2 / (numerador_1 + numerador_2 + numerador_3)

        pi_3 = numerador_3 / (numerador_1 + numerador_2 + numerador_3)

        estim.weights_ = [pi_1, pi_2, pi_3]

        avg_log_likelihood_act = estim.score(samples)
        if avg_log_likelihood_act < avg_log_likelihood_prev:
            break
        avg_log_likelihood_prev = avg_log_likelihood_act

    return (
        abs(estim.weights_[0] - 0.4) + abs(estim.weights_[1] - 0.6) + estim.weights_[2]
    )


if __name__ == "__main__":
    n_experiments = 50
    samples, _ = mixtura_original.sample(1000)
    r_1000_sinregu = (
        sum([experiment(samples, 0) for u in tqdm(range(n_experiments))])
        / n_experiments
    )
    r_100_regu = (
        sum([experiment(samples, 0.1) for u in tqdm(range(n_experiments))])
        / n_experiments
    )

    samples, _ = mixtura_original.sample(50)
    r_50_sinregu = (
        sum([experiment(samples, 0) for u in tqdm(range(n_experiments))])
        / n_experiments
    )
    r_50_regu = (
        sum([experiment(samples, 0.1) for u in tqdm(range(n_experiments))])
        / n_experiments
    )

    print(f"50 Samples without regularization {r_50_sinregu:.4f}")
    print(f"50 Samples with regularization {r_50_regu:.4f}")
    print(f"1000 Samples without regularization {r_1000_sinregu:.4f}")
    print(f"1000 Samples with regularization {r_100_regu:.4f}")
