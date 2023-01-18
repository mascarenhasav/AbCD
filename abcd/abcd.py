'''
Code for an Adaptative Dynamic Particle Swarm Optimization algorithm
using DEAP library.

Based on:
"T. Blackwell and J. Branke, “Multiswarms, exclusion, and anticonvergence
in dynamic environments,” IEEE Transactions on Evolutionary Computation,
vol. 10, no. 4, pp. 459–472, 2006."
and
"D. Yazdani, B. Nasiri, A. Sepas-Moghaddam, and M. R. Meybodi, “A novel
multi-swarm algorithm for optimization in dynamic environments based on
particle swarm optimization,” Applied Soft Computing, vol. 13, no. 04,
pp. 2144–2158, 2013."

Alexandre Mascarenhas

2023/1
'''

import json
import shutil
import itertools
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import os
import csv
import ast
import sys
import getopt
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.benchmarks import movingpeaks
import time



# datetime variables
cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

# Global variables
nevals = 0 # Number of evaluations
run = 0 # Current running
peaks = 0
env = 0
changesEnv = [0 for _ in range(100)]
path = ""

'''
Create the particle, with its initial position and speed
being randomly generated
'''
def ale(seed, min, max):
    return random.uniform(min, max)
def generate(ndim, pmin, pmax, smin, smax):
    part = creator.Particle(ale(i, pmin, pmax) for i in range(ndim))
    part.speed = [ale(i, smin, smax) for i in range(ndim)]
    part.smin = smin
    part.smax = smax
    return part

'''
Update the position of the particles
'''
def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


'''
Define the functions responsibles to create the objects of the algorithm
particles, swarms, the update and evaluate function also
'''
def createToolbox(parameters):
    BOUNDS_POS = parameters["BOUNDS_POS"]
    BOUNDS_VEL = parameters["BOUNDS_VEL"]
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, ndim=parameters["NDIM"],\
    pmin=BOUNDS_POS[0], pmax=BOUNDS_POS[1], smin=BOUNDS_VEL[0],smax=BOUNDS_VEL[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=parameters["phi1"], phi2=parameters["phi2"])
    toolbox.register("evaluate", evaluate)
    return toolbox

'''
Write the log of the algorithm over the generations on a csv file
'''
def writeLog(mode, filename, header, data=None):
    if(mode==0):
        # Create csv file
        with open(filename, "w") as file:
            csvwriter = csv.DictWriter(file, fieldnames=header)
            csvwriter.writeheader()
    elif(mode==1):
        # Writing in csv file
        with open(filename, mode="a") as file:
            csvwriter = csv.DictWriter(file, fieldnames=header)
            csvwriter.writerows(data)
           # for i in range(len(data)):
           #     csvwriter.writerows(data[i])

'''
Fitness function. Returns the error between the fitness of the particle
and the global optimum
'''
def evaluate(x, function, parameters):
    global nevals
    fitInd = function(x)[0]
    globalOP = function.maximums()[0][0]
    fitness = [abs( fitInd - globalOP )]
    nevals += 1
    if(parameters["CHANGE"]):
        changeEnvironment(function, parameters)
    return fitness


'''
Write the position and fitness of the peaks on
the 'optima.csv' file. The number of the peaks will be
NPEAKS_MPB*NCHANGES
'''
def saveOptima(parameters, mpb, path):
    opt = [0 for _ in range(parameters["NPEAKS_MPB"])]
    for i in range(parameters["NPEAKS_MPB"]):
        opt[i] = mpb.maximums()[i]
    with open(f"{path}/optima.csv", "a") as f:
        write = csv.writer(f)
        write.writerow(opt)

'''
Check if the dirs already exist, and if not, create them
Returns the path
'''
def checkDirs(path):
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    path += f"/{year}-{month}-{day}"
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    path += f"/{hour}-{minute}"
    if(os.path.isdir(path) == False):
        os.mkdir(path)
    return path

'''
Check if the dirs already exist, and if not, create them
Returns the path
'''
def antiConvergence(pop, parameters, randomInit):
    nconv = 0
    if(parameters["RCONV"]):
        rconv = parameters["RCONV"]
    else:
        rconv  = 0.75*(parameters["BOUNDS_POS"][1] - parameters["BOUNDS_POS"][0]) \
                 / (2 * len(pop)**(1.0/parameters["NDIM"]))
    wswarmId = None
    wswarm = None
    for i, swarm in enumerate(pop):
        # Compute the diameter of the swarm
        for p1, p2 in itertools.combinations(swarm, 2):
            d = math.sqrt(sum((x1 - x2)**2. for x1, x2 in zip(p1, p2)))
            if d > 2*rconv:
                nconv += 1
            # Search for the worst swarm according to its global best
            if not wswarm or swarm.best.fitness < wswarm.best.fitness:
                wswarmId = i
                wswarm = swarm
            break

    # If all swarms have converged, remember to randomize the worst
    if nconv == 0:
        randomInit[wswarmId] = 1

    return randomInit

'''
Check if the dirs already exist, and if not, create them
Returns the path
'''
def exclusion(pop, parameters, randomInit):
    if(parameters["REXCL"]):
        rexcl = parameters["REXCL"]
    else:
        rexcl  = (parameters["BOUNDS_POS"][1] - parameters["BOUNDS_POS"][0]) \
                 / (2 * len(pop)**(1.0/parameters["NDIM"]))
    reinit_swarms = set()
    for s1, s2 in itertools.combinations(range(len(pop)), 2):
        # Swarms must have a best and not already be set to reinitialize
        if pop[s1].best and pop[s2].best and not (randomInit[s1] or randomInit[s2]):
            dist = 0
            for x1, x2 in zip(pop[s1].best, pop[s2].best):
                dist += (x1 - x2)**2.
            dist = math.sqrt(dist)
            if dist < rexcl:
                if pop[s1].best.fitness <= pop[s2].best.fitness:
                    randomInit[s1] = 1
                else:
                    randomInit[s2] = 1

    return randomInit


'''
Apply ES on the particle
'''
def ES_particle(part, sbest, parameters, P=1):
    if(parameters["RCLOUD"]):
        rcloud = parameters["RCLOUD"]
    else:
        rcloud  = 0.5*parameters["MOVE_SEVERITY_MPB"]
    for i in range(parameters["NDIM"]):
        part[i] = sbest[i] + P*(random.uniform(-1, 1)*rcloud)
    return part


'''
Apply LS on the best
'''
def localSearch(best, toolbox, parameters, mpb):
    if(parameters["RLS"]):
        rls = parameters["RLS"]
    else:
        rls  = 0.5*parameters["MOVE_SEVERITY_MPB"]
    bp = creator.Particle(best)
    for _ in range(parameters["ETRY"]):
        for i in range(parameters["NDIM"]):
            bp[i] = bp[i] + random.uniform(-1, 1)*rls
        bp.fitness.values = toolbox.evaluate(bp, mpb, parameters=parameters)
        if bp.fitness > best.fitness:
            best = creator.Particle(bp)
            best.fitness.values = bp.fitness.values
    return best


'''
Check if a change occurred in the environment
'''
def changeDetection(swarm, toolbox, mpb, parameters):
    change = 0
    sensor = toolbox.evaluate(swarm.best, mpb, parameters=parameters)
    if(sensor[0] != swarm.best.fitness.values[0]):
        #print(f"[CHANGE] nevals: {nevals}  sensor: {sensor}  sbest:{swarm.best.fitness.values[0]}")
        swarm.best.fitness.values = sensor
        change = 1
    return change


'''
Reevaluate each particle attractor and update swarm best
If ES_CHANGE_OP is activated, the position of particles is
changed by ES strategy
'''
def reevaluateSwarm(swarm, best, toolbox, mpb, parameters):
    for part in swarm:

        if(parameters["ES_CHANGE_OP"]):
            part = ES_particle(part, swarm.best, parameters)
            part.best = part

        part.best.fitness.values = toolbox.evaluate(part.best, mpb, parameters=parameters)
        if not swarm.best or swarm.best.fitness < part.best.fitness:
            swarm.best = creator.Particle(part.best)
            swarm.best.fitness.values = part.best.fitness.values

    return swarm, best


'''
Change the environment if nevals reach the defined value
'''
def changeEnvironment(mpb, parameters):
    # Change environment
    global changesEnv
    global nevals
    global peaks
    global env
    global path
    if(nevals in changesEnv):
        mpb.changePeaks() # Change the environment
        #env += 1
        if(peaks <= parameters["NCHANGES"]): # Save the optima values
            saveOptima(parameters, mpb, path)
            peaks += 1
        #print(f"MUDOU nevals: {nevals}")


'''
Algorithm
'''
def adpso(parameters, seed):
    startTime = time.time()
    # Create the DEAP creators
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None, bestfit=creator.FitnessMax)
    creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

    global nevals
    global run
    global peaks
    global changesEnv
    global path
    global env

    # Set the general parameters
    NEVALS = parameters["NEVALS"]
    POPSIZE = parameters["POPSIZE"]
    NSWARMS = parameters["NSWARMS"]
    SWARMSIZE = int(POPSIZE/NSWARMS)
    RUNS = parameters["RUNS"]
    debug = parameters["DEBUG"]

    # Setup of MPB
    scenario = movingpeaks.SCENARIO_1
    severity = parameters["MOVE_SEVERITY_MPB"]
    scenario["period"] = parameters["PERIOD_MPB"]
    scenario["npeaks"] = parameters["NPEAKS_MPB"]
    scenario["uniform_height"] = parameters["UNIFORM_HEIGHT_MPB"]
    scenario["move_severity"] = severity
    scenario["min_height"] = parameters["MIN_HEIGHT_MPB"]
    scenario["max_height"] = parameters["MAX_HEIGHT_MPB"]
    scenario["min_coord"] = parameters["MIN_COORD_MPB"]
    scenario["max_coord"] = parameters["MAX_COORD_MPB"]
    rcloud = severity
    rls = severity
    filename = f"{path}/{parameters['FILENAME']}"

    # Headers of the log files
    header = ["run", "gen", "nevals", "swarmId", "partId", "part", "partError", "sbest", "sbestError", "best", "bestError", "env"]
    writeLog(mode=0, filename=filename, header=header)
    headerOPT = [f"opt{i}" for i in range(parameters["NPEAKS_MPB"])]
    writeLog(mode=0, filename=f"{path}/optima.csv", header=headerOPT)

    # Create the toolbox functions
    toolbox = createToolbox(parameters)

    # Check if the changes should be random or pre defined
    if(parameters["RANDOM_CHANGES"]):
        changesEnv = [random.randint(parameters["RANGE_NEVALS_CHANGES"][0], parameters["RANGE_NEVALS_CHANGES"][1]) for _ in range(parameters["NCHANGES"])]
    else:
        changesEnv = parameters["CHANGES_NEVALS"]

    # Main loop of ITER runs
    for run in range(1, RUNS+1):
        random.seed(run**5)
        best = None
        nevals = 0
        env = 1
        change = 0
        gen = 1
        flagEnv = 0
        randomInit = [0 for _ in range(1, NSWARMS+2)]
        ES_particles = [i for i in range(1, int(parameters["ES_PARTICLE_PERC"]*SWARMSIZE)+1)]

        # Initialize the benchmark for each run with seed being the minute
        rndMPB = random.Random()
        rndMPB.seed(int(seed)**5)

        mpb = movingpeaks.MovingPeaks(dim=parameters["NDIM"], random=rndMPB, **scenario)

        # Create the population with NSWARMS of size SWARMSIZE
        pop = [toolbox.swarm(n=SWARMSIZE) for _ in range(NSWARMS)]

        # Save the optima values
        if(peaks <= parameters["NCHANGES"]):
            saveOptima(parameters, mpb, path)
            peaks += 1

        # Repeat until reach the number of evals
        while nevals < NEVALS+1:

            # Anti-convergence operator
            if(parameters["ANTI_CONVERGENCE_OP"] and gen > 2):
                randomInit = antiConvergence(pop, parameters, randomInit)

            # Exclusion operator
            if(parameters["EXCLUSION_OP"] and gen > 2):
                randomInit = exclusion(pop, parameters, randomInit)

            # Local search operator
            if(parameters["LOCAL_SEARCH_OP"] and gen > 2):
                best = localSearch(best, toolbox, parameters, mpb)

            # PSO
            for swarmId, swarm in enumerate(pop, 1):

                # Change detection
                if(parameters["CHANGE_DETECTION_OP"] and swarm.best):
                    change = changeDetection(swarm, toolbox, mpb, parameters=parameters)

                if(change and swarm):
                    swarm, best = reevaluateSwarm(swarm, best, toolbox, mpb, parameters=parameters)
                    best = None
                    #print(f"[CHANGE] sbest:{swarm.best.fitness.values[0]}")
                    flagEnv += 1
                    if(flagEnv==NSWARMS):
                        env += 1
                        flagEnv = 0
                    randomInit[swarmId] = 0

                for partId, part in enumerate(swarm, 1):
                    if(gen > 2):
                        # If convergence or exclusion, randomize the particle
                        if(randomInit[swarmId]):
                            part = toolbox.particle()
                        else:
                            # ES Particle operator
                            #if(parameters["ES_PARTICLE_OP"]):
                            if(partId in ES_particles):
                                part = ES_particle(part, swarm.best, parameters)
                            else:
                                #print("AEEEE PSO")
                                toolbox.update(part, swarm.best)

                    # Evaluate the particle
                    part.fitness.values = toolbox.evaluate(part, mpb, parameters=parameters)

                    # Check if the particles are the best of itself and best at all
                    if not part.best or part.best.fitness < part.fitness:
                        part.best = creator.Particle(part)
                        part.best.fitness.values = part.fitness.values
                    if not swarm.best or swarm.best.fitness < part.fitness:
                        swarm.best = creator.Particle(part)
                        swarm.best.fitness.values = part.fitness.values
                    if not best or best.fitness < part.fitness:
                        best = creator.Particle(part)
                        best.fitness.values = part.fitness.values

                    # Save the log
                    #log = [{"run": run, "gen": gen, "nevals":nevals, "swarmId": swarmId, "partId": partId, "part":part, "partError": part.best.fitness.values[0], "sbest": swarm.best, "sbestError": swarm.best.fitness.values[0], "best": best, "bestError": best.fitness.values[0], "env": env}]
                    #writeLog(mode=1, filename=filename, header=header, data=log)

                    if(debug):
                        print(log)

                # Randomization complete
                randomInit[swarmId] = 0


            #Save the log
            log = [{"run": run, "gen": gen, "nevals":nevals, "best": best, "bestError": best.fitness.values[0], "env": env}]
            writeLog(mode=1, filename=filename, header=header, data=log)
            gen += 1

        #writeLog(mode=1, filename=filename, header=header, data=log)
        if(True):
            print(f"[RUN:{run:02}][GEN:{gen:03}][NEVALS:{nevals:05}] Best:{best.fitness.values[0]:.4f}")

    # Copy the config.ini file to the experiment dir
    #shutil.copyfile("config.ini", f"{path}/config.ini")
    executionTime = (time.time() - startTime)
    print(f"File generated: {path}/data.csv \nThx!")
    print('Execution time in seconds: ' + str(executionTime))

def main():
    global path
    seed = minute
    arg_help = "{0} -s <seed> -p <path>".format(sys.argv[0])
    path = "."

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:p:", ["help", "seed=", "path="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-s", "--seed"):
            seed = arg
        elif opt in ("-p", "--path"):
            path = arg

    print('seed:', seed)
    print('path:', path)
    # Read the parameters from the config file
    with open(f"{path}/config.ini") as f:
        parameters = json.loads(f.read())
    debug = parameters["DEBUG"]
    if(True):
        print("Parameters:")
        print(parameters)

    if path == ".":
        path = f"{parameters['PATH']}/{parameters['ALGORITHM']}"
        path = checkDirs(path)

    # Call the algorithm
    adpso(parameters, seed)

    # For automatic calling of the plot functions
    if(parameters["PLOT"]):
        os.system(f"python3 ../../PLOTcode.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")
        os.system(f"python3 ../../PLOT2code.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")


if __name__ == "__main__":
    main()


