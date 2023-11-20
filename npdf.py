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
    idx: int= 0
    jdx: int= 0
    lower: float
    mu: float
    sigma: float
    upper: float
    mu_space: np.ndarray
    sigma_space: np.ndarray
    variance: np.ndarray
    hist: Tuple[np.ndarray, np.ndarray]
    naive_minima: Tuple[Tuple[int, int], Tuple[float, float, float]]
    ax: Axes3D
    fig: Figure
    def __init__(self, bins: int, data: pd.Series) -> None:
        self.bins = bins
        self.data: pd.Series= data
        self.mu, self.sigma = norm.fit(data)
        self.lower, self.upper = np.min(data), np.max(data)
        self.x: np.ndarray= np.linspace(self.lower, self.upper, bins)
        self.prob_density: np.ndarray= norm.pdf(self.x, self.mu, self.sigma)
    def plot_histogram(self):
        self.hist = plt.hist(self.data,
            bins=self.bins, density=True)
        plt.plot(self.x, self.prob_density, 'k', linewidth=2)
        plt.axvline(self.mu, color='red')
        plt.axvline(self.mu + self.sigma, color='gray')
        plt.axvline(self.mu - self.sigma, color='gray')
        plt.title(" ".join(("Probability Density of {} VS Normal Model",
            "Given mu={:.2f}, sigma={:.2f}"))
            .format(self.data.name, self.mu, self.sigma))
        plt.show()
    def get_variance(self) -> float:
        mu: float= self.mu_space[self.idx][self.jdx]
        sigma: float= self.sigma_space[self.idx][self.jdx]
        self.variance[self.idx][self.jdx] = np.sum(
            (norm.pdf(self.x, mu, sigma) - self.prob_density)**2)
    def update_minima(self) -> bool:
        i, j = self.idx, self.jdx
        if self.variance[i][j] < self.naive_minima[1][2]:
            self.naive_minima = (i, j), (self.mu_space[i][j],
            self.sigma_space[i][j],
            self.variance[i][j])        
    def generate_data(self):
        self.idx: int= 0
        self.jdx: int= 0
        self.variance: np.ndarray= np.empty([self.bins, self.bins], dtype=float)
        self.mu_space: np.ndarray= self.x.copy()
        self.sigma_space: np.ndarray= np.linspace(0.5, 2*self.sigma, self.bins)
        self.mu_space, self.sigma_space = np.meshgrid(self.mu_space,
            self.sigma_space)
        self.get_variance()
        self.naive_minima = (self.idx, self.jdx), (self.mu_space[self.idx][self.jdx],
            self.sigma_space[self.idx][self.jdx],
            self.variance[self.idx][self.jdx])
        self.jdx += 1
        while self.idx < self.bins:
            while self.jdx < self.bins:
                self.get_variance()
                self.update_minima()
                self.jdx += 1
            self.jdx = 0
            self.idx += 1
        self.variance = np.ma.masked_where(~np.isfinite(self.variance), self.variance)
    def plot_parameter_space(self):
        self.generate_data()
        title: str= "\n".join(("Contour and Surface of Probability Density Variance",
            "as a Function of mu, sigma Plane",
            "Naive Minimum Variance at ({:.2f}, {:.2f}, {:.2f})"))
        min_x, min_y, min_z = self.naive_minima[1]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.plot_surface(self.mu_space, self.sigma_space, self.variance, cmap=cm.cubehelix_r,
            rstride=1, cstride=1, alpha=0.5)
        self.ax.contour(self.mu_space, self.sigma_space, self.variance, 10, cmap=cm.cubehelix_r,
            linestyles="solid", offset=0)
        self.ax.contour(self.mu_space, self.sigma_space, self.variance, 10, colors="k",
            linestyles="solid")
        self.ax.scatter3D(min_x, min_y, min_z)
        self.ax.set_xlabel("mu")
        self.ax.set_ylabel("sigma")
        self.ax.set_zlabel("Variance")
        self.ax.grid(visible=False)

        plt.title(title.format(min_x, min_y, min_z))
        plt.show()
