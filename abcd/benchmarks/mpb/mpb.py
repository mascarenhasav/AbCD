from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import json


#plt.rcParams["figure.figsize"] = (20, 8)
#plt.style.use("dark_background")
#plt.rcParams["axes.facecolor"] = "#1c1c1c"
plt.rcParams["axes.facecolor"] = "cornsilk"
plt.rcParams["axes.facecolor"] = "white"
#plt.rcParams["savefig.facecolor"] = "#1c1c1c"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams['grid.color'] = "dimgrey"


try:
    import numpy as np
except:
    exit()

import random
rnd = random.Random()
rnd.seed(128)

from deap.benchmarks import movingpeaks

# reading the parameters from the config file
with open("./config.ini") as f:
    parameters = json.loads(f.read())
debug = parameters["DEBUG"]
if(debug):
    print("Parameters:")
    print(parameters)

# Setup of MPB
scenario = movingpeaks.SCENARIO_1
scenario["period"] = parameters["PERIOD_MPB"]
scenario["npeaks"] = parameters["NPEAKS_MPB"]
scenario["uniform_height"] = parameters["UNIFORM_HEIGHT_MPB"]
scenario["move_severity"] = parameters["MOVE_SEVERITY_MPB"]
scenario["min_height"] = parameters["MIN_HEIGHT_MPB"]
scenario["max_height"] = parameters["MAX_HEIGHT_MPB"]
mpb = movingpeaks.MovingPeaks(dim=parameters["NDIM"],  random=rnd, **scenario)

for i in range (parameters["NCHANGES"]):
    fig = plt.figure()
    ax = Axes3D(fig)

    '''
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.tick_params(axis="x",colors="white")
    ax.tick_params(axis="y",colors="white")
    ax.tick_params(axis="z",colors="white")
    '''

    ax.xaxis.label.set_color("dimgrey")
    ax.yaxis.label.set_color("dimgrey")
    ax.zaxis.label.set_color("dimgrey")
    ax.tick_params(axis="x",colors="dimgrey")
    ax.tick_params(axis="y",colors="dimgrey")
    ax.tick_params(axis="z",colors="dimgrey")

    #ax.w_xaxis.set_pane_color((0.28,0.28,0.28,0.28))
    ax.w_xaxis.set_pane_color((0.28,0.28,0.28,0))
    ax.w_yaxis.set_pane_color((0.28,0.28,0.28,0))
    ax.w_zaxis.set_pane_color((0.28,0.28,0.28,0))

    X = np.arange(0, 100, 1.0)
    Y = np.arange(0, 100, 1.0)
    X, Y = np.meshgrid(X, Y)
    Z = np.fromiter(map(lambda x: mpb(x)[0], zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.grid(1)
    plt.grid(which="major", color="dimgrey", linewidth=0.8)
    plt.grid(which="minor", color="dimgrey", linestyle=":", linewidth=0.5)
    plt.savefig(f"mpb3-white{i}.png", format="png")
    plt.show()
    mpb.changePeaks()

#Z = np.fromiter(map(lambda x: mpb(x)[0], zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)


