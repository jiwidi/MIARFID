import numpy as np
from sklearn import mixture

regul_param = 2

samples = np.random.randint(10, size=(1000, 1))
estim = mixture.GaussianMixture(
    n_components=3,
    covariance_type="spherical",
    weights_init=np.array([0.7, 0.1, 0.2]),
    means_init=np.array([[-6], [2], [0]]),
    precisions_init=np.array([0.25, 0.25, 0.25]),
)
estim.covariances_ = np.array([4, 4, 4])
estim.means_ = np.array([[-6], [2], [0]])
estim.weights_ = np.array([0.7, 0.1, 0.2])
estim.precisions_cholesky_ = np.array([0.5, 0.5, 0.5])

estim.fit(samples)
avg_log_likelihood_prev = 10e9

# 1000 iteraciones máximo
for i in range(1, 1000):
    probs_act = estim.predict_proba(samples)
    numerador_1 = 0
    numerador_2 = 0
    numerador_3 = 0
    for p in probs_act:
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
