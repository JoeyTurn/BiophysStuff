import numpy as np
import math

class Density():

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.dens = np.zeros((self.height, self.width))  # adds inner_factor density to points inside boundries

    @property
    def get_density(self) -> np.ndarray:
        return self.dens

    def pointDensity(self, x, y, inc) -> np.ndarray:
        self.dens[int(y), int(x)] += inc
        return self.dens

    def verticalLineDensity(self, width, bheight, sheight, inc) -> np.ndarray:
        for i in range(int(sheight-bheight)):
            self.dens[int(sheight) + i][int(width)] += inc
        return self.dens

    def boxDensity(self, bheight, bwidth, sheight, swidth, inc) -> np.ndarray:
        for i in range(int(bwidth)):
            for j in range(int(bheight)):
                self.dens[int(sheight) + j][int(swidth) + i] += inc
        return self.dens

    def semicircleDensity(self, radius: int, theta_i: int, theta_f: int, x: int, y: int, inc) -> np.ndarray:
        denscopy = np.zeros((self.height, self.width))
        for r in range(int(radius)):
            if r == 0:
                denscopy[int(y), int(x)] = inc
            else:
                for theta in range(theta_f-theta_i):
                    ang = theta + theta_i
                    xloc = r * math.cos(math.radians(ang)) + x
                    yloc = r * math.sin(math.radians(ang)) + y
                    denscopy[int(yloc), int(xloc)] = inc
        self.dens = self.dens+denscopy
        return self.dens