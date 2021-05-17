import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_continuous
from scipy.stats import norm


class MixtureModel(rv_continuous):
    def __init__(self, submodels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels

    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x)
        for submodel in self.submodels[1:]:
            pdf += submodel.pdf(x)
        pdf /= len(self.submodels)
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.randint(len(self.submodels), size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


if __name__ == "__main__":
    mixture_gaussian_model = MixtureModel([norm(-3, 1), norm(3, 1)])
    x_axis = np.arange(-6, 6, 0.001)
    mixture_pdf = mixture_gaussian_model.pdf(x_axis)
    mixture_rvs = mixture_gaussian_model.rvs(10000)
    plt.hist(mixture_rvs, bins=100, density=True)
    plt.show()

