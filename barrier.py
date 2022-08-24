import numpy as np
import math

class Barrier:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.barrier = np.zeros((self.height, self.width), bool)  # True wherever there's a barrier

    @property
    def get_barrier(self) -> np.ndarray:
        return self.barrier

    def check_bool(self, a, b) -> bool:
        self.bool_check = not self.barrier[int(a), int(b)]
        return self.bool_check

    def point(self, x_point, y_point) -> np.ndarray:
        self.barrier[int(y_point), int(x_point)] = self.check_bool(int(y_point), int(x_point))
        return self.barrier

    #barrier[y_min:y_max, x_min:x_max]
    def vertical_line(self, width, start_height, end_height=None, length=None) -> np.ndarray:
        self.check_bool(start_height + 1, width)
        if end_height is not None:
            self.barrier[int(start_height):int(end_height)+1, int(width)] = self.bool_check
        else:
            try:
                self.barrier[int(start_height):int(start_height+length+1), int(width)] = self.bool_check
            except:
                pass
        return self.barrier

    def horizontal_line(self, height, start_width, end_width=None, length=None) -> np.ndarray:
        self.check_bool(height, start_width)
        if end_width is not None:
            self.barrier[int(height), int(start_width):int(end_width)] = self.bool_check
        else:
            try:
                self.barrier[int(height), int(start_width):start_width + length] = self.bool_check
            except:
                pass
        return self.barrier

    def box(self, start_height, end_height, start_width, end_width) -> np.ndarray:
        self.barrier = self.horizontal_line(start_height, start_width, end_width)
        self.barrier = self.horizontal_line(end_height, start_width, end_width)
        self.barrier = self.vertical_line(start_width, start_height, end_height)
        self.barrier = self.vertical_line(end_width, start_height, end_height)
        return self.barrier

    #theta = 0 corresponds to -pi rad
    def semicircle(self, radius, theta_i: int, theta_f: int, x_point, y_point) -> np.ndarray:
        self.bool_check = self.check_bool(y_point + radius-1, x_point + radius-1)
        for theta in range(theta_f-theta_i):
            ang = theta + theta_i
            xloc = radius * math.cos(math.radians(ang)) + x_point-1
            yloc = radius * math.sin(math.radians(ang)) + y_point-1
            self.barrier[int(round(yloc)), int(round(xloc))] = self.bool_check
        return self.barrier

    def circle(self, radius, x_point, y_point) -> np.ndarray:
        return self.semicircle(radius, 0, 360, x_point, y_point)