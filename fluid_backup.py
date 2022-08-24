#https://stackoverflow.com/questions/29832055/animated-subplots-using-matplotlib

#imports
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation
from density import Density
from barrier import Barrier

# Define constants: ------------------------------------

# lattice dimensions
height = 100
width = 250

viscosity = 0.02  # fluid viscosity
omega = 1 / (3 * viscosity + 0.5)  # "relaxation" parameter
u0 = .01  # initial and in-flow speed
performanceData = True

window = np.ones((height, width))

# Define functions for later use: ----------------------

# macroscopic density
def find_rho(n, nN, nS, nE, nNE, nSE, nNW, nSW):
    return n + nN + nS + nE + nW + nNE + nSE + nNW + nSW

# macroscopic x velocity
def find_ux(nE, nNE, nSE, nW, nNW, nSW, rho):
    return (nE + nNE + nSE - nW - nNW - nSW) / rho

# macroscopic y velocity
def find_uy(nN, nNE, nNW, nS, nSE, nSW, rho):
    return (nN + nNE + nNW - nS - nSE - nSW) / rho

#to find neighbors of points on array
def rolling(arr: np.ndarray, dir) -> np.ndarray:
    match dir:
        case "N":
            return np.roll(arr, 1, axis=0)
        case "S":
            return np.roll(arr, -1, axis=0)
        case "E":
            return np.roll(arr, 1, axis=1)
        case "W":
            return np.roll(arr, -1, axis=1)
        case "NE":
            return rolling(rolling(arr, "N"), "E")
        case "NW":
            return rolling(rolling(arr, "N"), "W")
        case "SE":
            return rolling(rolling(arr, "S"), "E")
        case "SW":
            return rolling(rolling(arr, "S"), "W")

# stream 1 frame
def stream():
    global nN, nS, nE, nW, nNE, nNW, nSE, nSW
    nN = rolling(nN, "N")
    nS = rolling(nS, "S")
    nE = rolling(nE, "E")
    nW = rolling(nW, "W")
    nNE = rolling(nNE, "NE")
    nNW = rolling(nNW, "NW")
    nSE = rolling(nSE, "SE")
    nSW = rolling(nSW, "SW")

    #reflection
    nN[rolling(barrier, "N")] = nS[barrier]
    nS[rolling(barrier, "S")] = nN[barrier]
    nE[rolling(barrier, "E")] = nW[barrier]
    nW[rolling(barrier, "W")] = nE[barrier]
    nNE[rolling(barrier, "NE")] = nSW[barrier]
    nNW[rolling(barrier, "NW")] = nSE[barrier]
    nSE[rolling(barrier, "SE")] = nNW[barrier]
    nSW[rolling(barrier, "SW")] = nNE[barrier]

# "Images" for the animation
def curl(ux, uy):
    return rolling(uy, "W") - rolling(uy, "E") - rolling(ux, "S") + rolling(ux, "N")

def density(ux, uy):
    return (ux**2 + uy**2)**.5

# Collide particles within each cell to redistribute velocities (could be optimized a little more):
def collide():
    global rho, ux, uy, n, nN, nS, nE, nW, nNE, nNW, nSE, nSW
    rho = find_rho(n, nN, nS, nE, nNE, nSE, nNW, nSW)
    ux = find_ux(nE, nNE, nSE, nW, nNW, nSW, rho)
    uy = find_uy(nN, nNE, nNW, nS, nSE, nSW, rho)
    uxy2 = 1 - 1.5 * (ux**2+uy**2)
    n = (1 - omega) * n + omega * 4/9 * rho * uxy2
    nN = (1 - omega) * nN + omega * 1/9 * rho * (uxy2 + 3 * uy + 4.5 * uy**2)
    nS = (1 - omega) * nS + omega * 1/9 * rho * (uxy2 - 3 * uy + 4.5 * uy**2)
    nE = (1 - omega) * nE + omega * 1/9 * rho * (uxy2 + 3 * ux + 4.5 * ux**2)
    nW = (1 - omega) * nW + omega * 1/9 * rho * (uxy2 - 3 * ux + 4.5 * ux**2)
    nNE = (1 - omega) * nNE + omega * 1/36 * rho * (uxy2 + 3 * (ux + uy) + 4.5 * (ux**2 + uy**2 + 2 * ux*uy))
    nNW = (1 - omega) * nNW + omega * 1/36 * rho * (uxy2 + 3 * (-ux + uy) + 4.5 * (ux**2 + uy**2 - 2 * ux*uy))
    nSE = (1 - omega) * nSE + omega * 1/36 * rho * (uxy2 + 3 * (ux - uy) + 4.5 * (ux**2 + uy**2 - 2 * ux*uy))
    nSW = (1 - omega) * nSW + omega * 1/36 * rho * (uxy2 + 3 * (-ux - uy) + 4.5 * (ux**2 + uy**2 + 2 * ux*uy))

    #detects if wall is poorly constructed
    #do later lmoa


    # Force steady rightward flow at ends (no need to set 0, N, and S components):
    #flow("W", u0, nE, nW, nNE, nSE, nNW, nSW)

def flow(dir, u0, nE, nW, nNE, nSE, nNW, nSW):
    match(dir):
        # creating rightwards flow from the left side of the screen
        case "E":
            nE[:, 0] = 1 / 9 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nW[:, 0] = 1 / 9 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNE[:, 0] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSE[:, 0] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNW[:, 0] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSW[:, 0] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
        case "W":
            nW[:, -1] = 1 / 9 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nE[:, -1] = 1 / 9 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNW[:, -1] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSW[:, -1] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNE[:, -1] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSE[:, -1] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
        case "N":
            nN[0, :] = 1 / 9 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nS[0, :] = 1 / 9 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNE[0, :] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNW[0, :] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSW[0, :] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSE[0, :] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
        case "S":
            nS[-1, :] = 1 / 9 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nN[-1, :] = 1 / 9 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSE[-1, :] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nSW[-1, :] = 1 / 36 * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNW[-1, :] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
            nNE[-1, :] = 1 / 36 * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)

def next_frame(arg):  #arg is the frame number
    global startTime
    if performanceData and (arg % 100 == 0) and (arg > 0):
        endTime = time.monotonic()
        print(f"{(100 / (endTime - startTime)):1.1f} frames per second")
        startTime = endTime
    frameName = "frame%04d.png"%arg
    plt.savefig(frameName)
    frameList.write(frameName + '\n')
    for step in range(20):  # adjust number of steps for smooth animation
        stream()
        collide()
    fluid_image_curl.set_array(curl(ux, uy))
    fluid_image_density.set_array(density(ux, uy))
#    line[0].set_data(fluid_image_curl, barrierImage)
#    line[1].set_data(fluid_image_density, barrierImage)

    #if arg == 5:
    #    barrier[bounds[5][0]:bounds[5][1], bounds[5][2]] = False

    return (fluid_image_density, barrierImage)  # return the figure elements to redraw

def construct_barrier(height, width):
    barrier_object.box((height / 2 - 2 * height / 5), (height - height / 2 + 2 * height / 5), (width / 2 - width / 5),
                       (width / 2 + width / 5))
    barrier_object.vertical_line(width / 2 - width / 5, (height / 2 - height / 10), (height / 2 + height / 10))
    barrier_object.semicircle(2 * height / 5, 0, 180, width / 2 + width / 5, height / 2 + 1)
    barrier_object.vertical_line(width / 2 + width / 5, height / 2 - 2 * height / 5 + 1,
                                 height / 2 + 2 * height / 5 - 1)
    # barrier_object.box(0, height-1, 0, width-1)
    global barrier
    barrier = barrier_object.barrier

def construct_density(height, width):
    density_object.boxDensity(4 * height / 5, 2 * width / 5, height / 10, width / 2 - width / 5, 20)
    density_object.boxDensity(height, width, 0, 0, 1)

# get barrier, density ---------------------------------

barrier_object = Barrier(height, width)
density_object = Density(height, width)

construct_barrier(height, width)
construct_density(height, width)

# Initialize steady flow
boons_u0 = density_object.dens*u0

#particle densities along 9 directions -----------------

n = 4/9 * (window - 1.5*boons_u0**2)
nN = 1/9 * (window - 1.5 *boons_u0 ** 2)
nS = 1/9 * (window - 1.5 *boons_u0 ** 2)
nE = 1/9 * (window + 3 *boons_u0 + 4.5 *boons_u0 ** 2 - 1.5 *boons_u0 ** 2)
nW = 1/9 * (window - 3 *boons_u0 + 4.5 *boons_u0 ** 2 - 1.5 *boons_u0 ** 2)
nNE = 1/36 * (window + 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
nSE = 1/36 * (window + 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
nNW = 1/36 * (window - 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
nSW = 1/36 * (window - 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
rho = find_rho(n, nN, nS, nE, nNE, nSE, nNW, nSW)
ux = find_ux(nE, nNE, nSE, nW, nNW, nSW, rho)
uy = find_uy(nN, nNE, nNW, nS, nSE, nSW, rho)

# -------------------------------------------------------

# Here comes the graphics and animation...
theFig = plt.figure(figsize=(10, 6))

fluid_image_curl = plt.imshow(curl(ux, uy), origin='lower', norm=plt.Normalize(-.1, .1),
                                      cmap=plt.get_cmap('jet'), interpolation='none')
fluid_image_density = plt.imshow(density(ux, uy), origin='lower', norm=plt.Normalize(-.1, .1),
                                      cmap=plt.get_cmap('jet'), interpolation='none')
# See http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps for other cmap options
bImageArray = np.zeros((height, width, 4), np.uint8)  # an RGBA image
bImageArray[barrier, 3] = 255  # set alpha=255 only at barrier sites
#bImageArray[]
barrierImage = plt.imshow(bImageArray, origin='lower', interpolation='none')


# Function called for each successive animation frame:
startTime = time.monotonic()

# create a figure with two subplots
#fig, (ax1, ax2) = plt.subplots(2,1)

# intialize two line objects (one in each axes)
#line1 = ax1.plot([], [], lw=2)
#line2 = ax2.plot([], [], lw=2, color='r')
#line = [line1, line2]

frameList = open('frameList.txt','w')		# file containing list of images (to make movie)

animate = matplotlib.animation.FuncAnimation(theFig, next_frame, frames=100, interval=10, blit=True)
plt.show()