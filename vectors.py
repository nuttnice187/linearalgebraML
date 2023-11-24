from math import sqrt
from typing import List
import numpy as np
import numpy.typing as npt
import math
from matplotlib.patches import FancyArrow
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Vector:
    """
    Object representation of a vector.
    """
    origin: npt.NDArray[np.float64]
    direction: npt.NDArray[np.float64]
    name: str
    magnitude: float
    def __init__(self, name: str, origin: List[float], direction: List[float]
        ) -> None:
        assert len(origin) == len(direction), ' '.join(("Origin and direction",
            "must have the same number of elements."))
        self.origin, self.direction, self.name = origin, direction, name
        self.magnitude = math.sqrt(np.sum(np.array(self.direction)**2))

class VectorPlot:
    fig: Figure
    arrows: List[FancyArrow]
    def __init__(self, vectors: List[Vector]):
        n: int= len(vectors)
        colors: npt.NDArray= plt.cm.rainbow(np.linspace(0, 1, n))
        i_hat: FancyArrow= plt.arrow(0, 0, 1, 0, color='k', head_width=0.1,
            head_length=0.1, length_includes_head=True, width=0.04)
        j_hat: FancyArrow= plt.arrow(0, 0, 0, 1, color='k', head_width=0.1,
            head_length=0.1, length_includes_head=True, width=0.04)
        self.arrows = [i_hat, j_hat]
        labels: List[str]= ['$\\mathbf{\\^i}$', '$\\mathbf{\\^j}$']
        for i, v in enumerate(vectors):
            assert len(v.origin) < 3, ' '.join(("Vectors can be no more than",
                "2-dimensional."))
            assert v.name not in labels, "Vectors must have unique names."
            x, y = v.origin
            dx, dy = v.direction
            arrow: FancyArrow= plt.arrow(x, y, dx, dy, color=colors[i],
                head_width=0.1, head_length=0.1, length_includes_head=True,
                width=0.04)
            self.arrows.append(arrow)
            labels.append(r"$\mathbf{%s}$" % v.name)
        self.fig = plt.figure(1)
        plt.legend(self.arrows, labels)
