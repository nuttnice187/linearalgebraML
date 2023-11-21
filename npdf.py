# Create Data for Contour Surface
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

class NormalProbabilityDensityModel:
    """
    Given the number of bins and pandas Series of numeric data, plot probability
    density histogram VS normal density function and performs naive local minima
    discovery of variance for normal probability density parameter space: the mu,
    sigma plane.
    """
    __idx: int
    __jdx: int
    lower: float
    mu: float
    sigma: float
    upper: float
    x: np.ndarray
    prob_density_model: np.ndarray
    mu_space: np.ndarray
    sigma_space: np.ndarray
    variance: np.ndarray
    hist: Tuple[np.ndarray, np.ndarray]
    mle: Tuple[Tuple[int, int], Tuple[float, float, float], np.ndarray]
    ax: Axes3D
    fig: Figure
    def __init__(self, bins: int, data: pd.Series) -> None:
        self.bins = bins
        self.data: pd.Series= data
        self.mu, self.sigma = norm.fit(data)
        self.lower, self.upper = np.min(data), np.max(data)
        self.x = np.linspace(self.lower, self.upper, bins)
        self.prob_density_model = norm.pdf(self.x, self.mu, self.sigma)
    def plot_histogram(self) -> None:
        if hasattr(self, 'mle'):
            estimation_type:str = "Maximum Likelihood Estimation Model"
            self.prob_density_model = self.mle[2]
        else:            
            estimation_type:str = "Arithmetic Mean Model"
        title = ("\n".join(("Probability Density of {} VS {}",
            "Given mu={:.2f}, sigma={:.2f}"))
            .format(self.data.name, estimation_type, self.mu, self.sigma))
        self.hist = plt.hist(self.data, bins=self.bins, density=True)
        plt.plot(self.x, self.prob_density_model, 'k', linewidth=2)
        plt.axvline(self.mu, color='red')
        plt.axvline(self.mu + self.sigma, color='gray')
        plt.axvline(self.mu - self.sigma, color='gray')
        plt.title(title)
        plt.show()
    def __get_variance(self) -> None:
        i, j = self.__idx, self.__jdx
        mu: float= self.mu_space[i][j]
        sigma = self.sigma_space[i][j]
        self.prob_density_model = norm.pdf(self.x, mu, sigma)
        prob_density_actual: np.ndarray= self.hist[0]
        self.variance[i][j] = np.sum(
            (self.prob_density_model - prob_density_actual)**2)
    def __update_minima(self) -> None:
        i, j = self.__idx, self.__jdx
        if self.variance[i][j] < self.mle[1][2]:
            self.mle = (i, j), (self.mu_space[i][j],
                self.sigma_space[i][j],
                self.variance[i][j]), self.prob_density_model
            self.mu = self.mu_space[i][j]
            self.sigma = self.sigma_space[i][j]
    def __generate_data(self) -> None:
        self.__idx = 0
        self.__jdx = 0
        self.variance = np.empty([self.bins, self.bins], dtype=float)
        self.mu_space = self.x.copy()
        self.sigma_space = np.linspace(0.5, 2*self.sigma, self.bins)
        self.mu_space, self.sigma_space = np.meshgrid(self.mu_space,
            self.sigma_space)
        self.__get_variance()
        self.mle = (self.__idx, self.__jdx), (
            self.mu_space[self.__idx][self.__jdx],
            self.sigma_space[self.__idx][self.__jdx],
            self.variance[self.__idx][self.__jdx]), norm.pdf(self.x,
                self.mu_space[self.__idx][self.__jdx],
                self.sigma_space[self.__idx][self.__jdx])
        self.__jdx += 1
        while self.__idx < self.bins:
            while self.__jdx < self.bins:
                self.__get_variance()
                self.__update_minima()
                self.__jdx += 1
            self.__jdx = 0
            self.__idx += 1
        self.variance = np.ma.masked_where(~np.isfinite(self.variance),
            self.variance)
    def fit(self) -> None:
        self.__generate_data()
        title: str= "\n".join((
            "Contour and Surface of Probability Density Variance",
            "as a Function of mu, sigma Plane",
            "Naive Minimum Variance at ({:.2f}, {:.2f}, {:.2E})"))
        min_x, min_y, min_z = self.mle[1]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.plot_surface(self.mu_space, self.sigma_space, self.variance,
            cmap=cm.cubehelix_r, rstride=1, cstride=1, alpha=0.5)
        self.ax.contour(self.mu_space, self.sigma_space, self.variance, 10,
            cmap=cm.cubehelix_r, linestyles="solid", offset=0)
        self.ax.contour(self.mu_space, self.sigma_space, self.variance, 10,
            colors="k", linestyles="solid")
        self.ax.scatter3D(min_x, min_y, min_z)
        self.ax.set_xlabel("mu")
        self.ax.set_ylabel("sigma")
        self.ax.set_zlabel("Variance")
        self.ax.grid(visible=False)

        plt.title(title.format(min_x, min_y, min_z))
        plt.show()
    def predict(self, x: float, mu: Optional[float]=None, sigma: Optional[float]=None
        ) -> float:
        if not (mu and sigma):
            assert hasattr(self, 'mle'), '\n'.join(("MLE does not exist.",
                "Execute fit() method, first, or provide mu and sigma."))
            mu, sigma = self.mu, self.sigma
        return norm.pdf(x, mu, sigma)
