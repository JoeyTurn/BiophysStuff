import random


class Destroyer():

    def __init__(self, height, width, n):
        self.height = height
        self.width = width
        self.population = n

    #10 degrees less than possible angles in order for hole to be large enough
    def simpleAngle(self):
        self.ang = int(random.random() * 360)
        while self.ang < 55 or self.ang > 305 or (self.ang > 125 and self.ang < 225):
            self.ang = int(random.random() * 360)
        return self.ang
