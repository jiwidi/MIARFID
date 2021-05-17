import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.stats as ss
from sklearn import mixture
import sklearn
from mixture import MixtureModel
from scipy.stats import norm
from tqdm import tqdm


mixture_gaussian_model = MixtureModel([norm(-6, 4), norm(2, 4)])
samples = mixture_gaussian_model.rvs(1000).reshape(-1, 1)
np.random.seed(17)

estim = mixture.GaussianMixture(
    n_components=3,
    covariance_type="spherical",
    weights_init=np.array([0.7, 0.1, 0.2]),
    means_init=np.array([[-6], [2], [0]]),
    precisions_init=np.array([0.25, 0.25, 0.25]),  # 1/4 inverse of covariance
    random_state=17,
)
regul_param = 0
estim.covariances_ = np.array([4, 4, 4])
estim.means_ = np.array([[-6], [2], [0]])
estim.weights_ = np.array([0.7, 0.1, 0.2])
estim.precisions_cholesky_ = np.array([0.5, 0.5, 0.5])

avg_log_likelihood_prev = -10e9

# print(
#     "Round \t%3d\t%3f\t%3f\t%3f"
#     % (0, estim.weights_[0], estim.weights_[1], estim.weights_[2])
# )
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
    # print(
    #     "Round \t%3d\t%3f\t%3f\t%3f"
    #     % (i, estim.weights_[0], estim.weights_[1], estim.weights_[2])
    # )
    avg_log_likelihood_act = estim.score(samples)
    if avg_log_likelihood_act < avg_log_likelihood_prev:
        break
    avg_log_likelihood_prev = avg_log_likelihood_act

print((abs(estim.weights_[0] - 0.4) + abs(estim.weights_[1] - 0.6) + estim.weights_[2]))

sys.exit(0)


def experiment(muestras=50, regul_param=0):
    mixture_gaussian_model = MixtureModel([norm(-6, 4), norm(2, 4)])
    samples = mixture_gaussian_model.rvs(muestras).reshape(-1, 1)
    np.random.seed(17)

    estim = mixture.GaussianMixture(
        n_components=3,
        covariance_type="spherical",
        weights_init=np.array([0.7, 0.1, 0.2]),
        means_init=np.array([[-6], [2], [0]]),
        precisions_init=np.array([0.25, 0.25, 0.25]),  # 1/4 inverse of covariance
        random_state=17,
    )

    estim.covariances_ = np.array([4, 4, 4])
    estim.means_ = np.array([[-6], [2], [0]])
    estim.weights_ = np.array([0.7, 0.1, 0.2])
    estim.precisions_cholesky_ = np.array([0.5, 0.5, 0.5])

    avg_log_likelihood_prev = -10e9

    # print(
    #     "Round \t%3d\t%3f\t%3f\t%3f"
    #     % (0, estim.weights_[0], estim.weights_[1], estim.weights_[2])
    # )
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
        # print(
        #     "Round \t%3d\t%3f\t%3f\t%3f"
        #     % (i, estim.weights_[0], estim.weights_[1], estim.weights_[2])
        # )
        avg_log_likelihood_act = estim.score(samples)
        if avg_log_likelihood_act < avg_log_likelihood_prev:
            break
        avg_log_likelihood_prev = avg_log_likelihood_act
    print(
        (
            abs(estim.weights_[0] - 0.4)
            + abs(estim.weights_[1] - 0.6)
            + estim.weights_[2]
        )
    )
    return (
        abs(estim.weights_[0] - 0.4) + abs(estim.weights_[1] - 0.6) + estim.weights_[2]
    )


if __name__ == "__main__":
    n_experiments = 50
    r_50_sinregu = (
        sum([experiment(50, 0) for u in tqdm(range(n_experiments))]) / n_experiments
    )
    r_50_regu = (
        sum([experiment(50, 0.1) for u in tqdm(range(n_experiments))]) / n_experiments
    )
    r_1000_sinregu = (
        sum([experiment(1000, 0) for u in tqdm(range(n_experiments))]) / n_experiments
    )
    r_100_regu = (
        sum([experiment(1000, 0.1) for u in tqdm(range(n_experiments))]) / n_experiments
    )

    print(f"50 Samples without regularization {r_50_sinregu:.2f}")
    print(f"50 Samples with regularization {r_50_regu:.2f}")
    print(f"1000 Samples without regularization {r_1000_sinregu:.2f}")
    print(f"1000 Samples with regularization {r_100_regu:.2f}")
