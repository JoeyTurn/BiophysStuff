# LatticeBoltzmannDemo.py:  a two-dimensional lattice-Boltzmann "wind tunnel" simulation
# Uses np to speed up all array handling.
# Uses matplotlib to plot and animate the curl of the macroscopic velocity field.

# Copyright 2013, Daniel V. Schroeder (Weber State University) 2013

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated data and documentation (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# Except as contained in this notice, the name of the author shall not be used in
# advertising or otherwise to promote the sale, use or other dealings in this
# Software without prior written authorization.

# Credits:
# The "wind tunnel" entry/exit conditions are inspired by Graham Pullan's code
# (http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml).  Additional inspiration from
# Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html).  Other portions of code are based
# on Wagner (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
# Gonsalves (http://www.physics.buffalo.edu/phy411-506-2004/index.html; code adapted from Succi,
# http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).

# For related materials see http://physics.weber.edu/schroeder/fluids

#https://stackoverflow.com/questions/29832055/animated-subplots-using-matplotlib

import numpy as np, time, matplotlib.pyplot as plt, matplotlib.animation

# Define constants:
height = 80  # lattice dimensions
width = 200
viscosity = 0.02  # fluid viscosity
omega = 1 / (3 * viscosity + 0.5)  # "relaxation" parameter
u0 = .01  # initial and in-flow speed
four9ths = 4.0 / 9.0  # abbreviations for lattice-Boltzmann weight factors
one9th = 1.0 / 9.0
one36th = 1.0 / 36.0
performanceData = True  # set to True if performance data is desired
inner_factor = 8

# Initialize all the arrays to steady rightward flow:
#n0 = four9ths * (np.ones((height, width)) - 1.5 * u0 ** 2)  # particle densities along 9 directions
boons = np.ones((height,width))					# adds inner_factor density to points inside boundries
for i in range(int(1*width/2)):
    for j in range(int(height)):
        boons[j][i] = 1+inner_factor

boons_u0 = boons*u0

n0 = four9ths * (np.ones((height,width)) - 1.5*boons_u0**2)	# particle densities along 9 directions

#n0l = four9ths * (np.ones((height, int(width/2))) - 1.5 * inner_factor * u0 ** 2)
#n0r = four9ths * (np.ones((height, int(width/2))) - 1.5 * u0 ** 2)
#n0 = np.concatenate([n0l, n0r], axis=1)
nN = one9th * (np.ones((height, width)) - 1.5 *boons_u0 ** 2)
nS = one9th * (np.ones((height, width)) - 1.5 *boons_u0 ** 2)
nE = one9th * (np.ones((height, width)) + 3 *boons_u0 + 4.5 *boons_u0 ** 2 - 1.5 *boons_u0 ** 2)
nW = one9th * (np.ones((height, width)) - 3 *boons_u0 + 4.5 *boons_u0 ** 2 - 1.5 *boons_u0 ** 2)
nNE = one36th * (np.ones((height, width)) + 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
nSE = one36th * (np.ones((height, width)) + 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
nNW = one36th * (np.ones((height, width)) - 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
nSW = one36th * (np.ones((height, width)) - 3 * boons_u0 + 4.5 * boons_u0 ** 2 - 1.5 * boons_u0 ** 2)
rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW  # macroscopic density
ux = (nE + nNE + nSE - nW - nNW - nSW) / rho  # macroscopic x velocity
uy = (nN + nNE + nNW - nS - nSE - nSW) / rho  # macroscopic y velocity

# Initialize barriers:
barrier = np.zeros((height, width), bool)  # True wherever there's a barrier
bounds = [[int(height-height), int(height/2)-1, int(width/2-width/5), 0],
          [int(height-height/2)+1, int(height-0), int(width/2-width/5), 0],] #y_min, y_max, x_min, x_max

barrier[bounds[0][0]:bounds[0][1], bounds[0][2]] = True			# simple linear barrier #2nd index starts at y = 0
barrier[bounds[1][0]:bounds[1][1], bounds[0][2]] = True

barrierN = np.roll(barrier, 1, axis=0)  # sites just north of barriers
barrierS = np.roll(barrier, -1, axis=0)  # sites just south of barriers
barrierE = np.roll(barrier, 1, axis=1)  # etc.
barrierW = np.roll(barrier, -1, axis=1)
barrierNE = np.roll(barrierN, 1, axis=1)
barrierNW = np.roll(barrierN, -1, axis=1)
barrierSE = np.roll(barrierS, 1, axis=1)
barrierSW = np.roll(barrierS, -1, axis=1)


# Move all particles by one step along their directions of motion (pbc):
def stream():
    global nN, nS, nE, nW, nNE, nNW, nSE, nSW
    nN = np.roll(nN, 1, axis=0)  # axis 0 is north-south; + direction is north
    nNE = np.roll(nNE, 1, axis=0)
    nNW = np.roll(nNW, 1, axis=0)
    nS = np.roll(nS, -1, axis=0)
    nSE = np.roll(nSE, -1, axis=0)
    nSW = np.roll(nSW, -1, axis=0)
    nE = np.roll(nE, 1, axis=1)  # axis 1 is east-west; + direction is east
    nNE = np.roll(nNE, 1, axis=1)
    nSE = np.roll(nSE, 1, axis=1)
    nW = np.roll(nW, -1, axis=1)
    nNW = np.roll(nNW, -1, axis=1)
    nSW = np.roll(nSW, -1, axis=1)
    # Use tricky boolean arrays to handle barrier collisions (bounce-back):
    nN[barrierN] = nS[barrier]
    nS[barrierS] = nN[barrier]
    nE[barrierE] = nW[barrier]
    nW[barrierW] = nE[barrier]
    nNE[barrierNE] = nSW[barrier]
    nNW[barrierNW] = nSE[barrier]
    nSE[barrierSE] = nNW[barrier]
    nSW[barrierSW] = nNE[barrier]


# Collide particles within each cell to redistribute velocities (could be optimized a little more):
def collide():
    global rho, ux, uy, n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW
    rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW
    ux = (nE + nNE + nSE - nW - nNW - nSW) / rho
    uy = (nN + nNE + nNW - nS - nSE - nSW) / rho
    ux2 = ux * ux  # pre-compute terms used repeatedly...
    uy2 = uy * uy
    u2 = ux2 + uy2
    omu215 = 1 - 1.5 * u2  # "one minus u2 times 1.5"
    uxuy = ux * uy
    n0 = (1 - omega) * n0 + omega * four9ths * rho * omu215
    nN = (1 - omega) * nN + omega * one9th * rho * (omu215 + 3 * uy + 4.5 * uy2)
    nS = (1 - omega) * nS + omega * one9th * rho * (omu215 - 3 * uy + 4.5 * uy2)
    nE = (1 - omega) * nE + omega * one9th * rho * (omu215 + 3 * ux + 4.5 * ux2)
    nW = (1 - omega) * nW + omega * one9th * rho * (omu215 - 3 * ux + 4.5 * ux2)
    nNE = (1 - omega) * nNE + omega * one36th * rho * (omu215 + 3 * (ux + uy) + 4.5 * (u2 + 2 * uxuy))
    nNW = (1 - omega) * nNW + omega * one36th * rho * (omu215 + 3 * (-ux + uy) + 4.5 * (u2 - 2 * uxuy))
    nSE = (1 - omega) * nSE + omega * one36th * rho * (omu215 + 3 * (ux - uy) + 4.5 * (u2 - 2 * uxuy))
    nSW = (1 - omega) * nSW + omega * one36th * rho * (omu215 + 3 * (-ux - uy) + 4.5 * (u2 + 2 * uxuy))
    # Force steady rightward flow at ends (no need to set 0, N, and S components):
    nE[:, 0] = one9th * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
    nW[:, 0] = one9th * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
    nNE[:, 0] = one36th * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
    nSE[:, 0] = one36th * (1 + 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
    nNW[:, 0] = one36th * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)
    nSW[:, 0] = one36th * (1 - 3 * u0 + 4.5 * u0 ** 2 - 1.5 * u0 ** 2)


# Compute curl of the macroscopic velocity field:
def curl(ux, uy):
    return np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1) - np.roll(ux, -1, axis=0) + np.roll(ux, 1, axis=0)
def density(ux, uy):
    return (ux**2 + uy**2)**.5

# Here comes the graphics and animation...
theFig = plt.figure(figsize=(10, 6))

fluid_image_curl = plt.imshow(curl(ux, uy), origin='lower', norm=plt.Normalize(-.1, .1),
                                      cmap=plt.get_cmap('jet'), interpolation='none')
fluid_image_density = plt.imshow(density(ux, uy), origin='lower', norm=plt.Normalize(-.1, .1),
                                      cmap=plt.get_cmap('jet'), interpolation='none')
# See http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps for other cmap options
bImageArray = np.zeros((height, width, 4), np.uint8)  # an RGBA image
bImageArray[barrier, 3] = 255  # set alpha=255 only at barrier sites
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
def next_frame(arg):  # (arg is the frame number, which we don't need)
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

    return (fluid_image_density, barrierImage)  # return the figure elements to redraw


animate = matplotlib.animation.FuncAnimation(theFig, next_frame, frames=100, interval=10, blit=True)
plt.show()