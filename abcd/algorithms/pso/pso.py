'''
Base code of the AbCD framework.

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
NCHANGES = 0

'''
Create the particle, with its initial position and speed
being randomly generated
'''
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
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
    toolbox.register("particle", generate, size=parameters["NDIM"],\
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
    if(parameters["BENCHMARK"] == "MPB"):
        globalOP = function.maximums()[0][0]
    elif(parameters["BENCHMARK"] == "H1"):
        globalOP = function([8.6998, 6.7665])[0]
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
def saveOptima(parameters, fitFunction, path):
    opt = [0]
    if(parameters["BENCHMARK"] == "MPB"):
        opt = [0 for _ in range(parameters["NPEAKS_MPB"])]
        for i in range(parameters["NPEAKS_MPB"]):
            opt[i] = fitFunction.maximums()[i]
    elif(parameters["BENCHMARK"] == "H1"):
        opt[0] = fitFunction([8.6998, 6.7665])[0]
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
Check if a change occurred in the environment
'''
def changeDetection(swarm, toolbox, fitFunction, change, parameters):
    sensor = toolbox.evaluate(swarm.best, fitFunction, parameters=parameters)
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
def reevaluateSwarm(swarm, toolbox, fitFunction, parameters):
    for part in swarm:
        if(parameters["ES_CHANGE_OP"]):
            part = ES_particle(part, swarm.best, parameters)
            part.best = part

        part.best.fitness.values = evaluate(part.best, fitFunction, parameters=parameters)
        if not swarm.best or swarm.best.fitness < part.best.fitness:
            swarm.best = creator.Particle(part.best)
            swarm.best.fitness.values = part.best.fitness.values

    return swarm


'''
Change the environment if nevals reach the defined value
'''
def changeEnvironment(fitFunction, parameters):
    # Change environment
    global changesEnv
    global nevals
    global peaks
    global path
    global NCHANGES
    if(nevals in changesEnv):
        fitFunction.changePeaks() # Change the environment
        if(peaks <= NCHANGES): # Save the optima values
            saveOptima(parameters, fitFunction, path)
            peaks += 1


'''
Algorithm
'''
def pso(parameters, seed):
    startTime = time.time()
    # Create the DEAP creators
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
               smin=None, smax=None, best=None, bestfit=creator.FitnessMin)
    creator.create("Swarm", list, best=None, bestfit=creator.FitnessMin)

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

    # Setup of MPB
    if (parameters["SCENARIO_MPB"] == 1):
        scenario = movingpeaks.SCENARIO_1
    elif (parameters["SCENARIO_MPB"] == 2):
        scenario = movingpeaks.SCENARIO_2
    elif (parameters["SCENARIO_MPB"] == 3):
        scenario = movingpeaks.SCENARIO_3

    severity = parameters["MOVE_SEVERITY_MPB"]
    scenario["period"] = 0
    scenario["npeaks"] = parameters["NPEAKS_MPB"]
    scenario["uniform_height"] = parameters["UNIFORM_HEIGHT_MPB"]
    scenario["move_severity"] = severity
    scenario["min_height"] = parameters["MIN_HEIGHT_MPB"]
    scenario["max_height"] = parameters["MAX_HEIGHT_MPB"]
    scenario["min_width"] = parameters["MIN_WIDTH_MPB"]
    scenario["max_width"] = parameters["MAX_WIDTH_MPB"]
    scenario["min_coord"] = parameters["MIN_COORD_MPB"]
    scenario["max_coord"] = parameters["MAX_COORD_MPB"]
    scenario["lambda_"] = parameters["LAMBDA_MPB"]
    rcloud = severity
    rls = severity
    filename = f"{path}/{parameters['FILENAME']}"

    # Headers of the log files
    if(parameters["LOG_ALL"]):
        header = ["run", "gen", "nevals", "swarmId", "partId", "part", "partError", "sbest", "sbestError", "best", "bestError", "Eo""env"]
    else:
        header = ["run", "gen", "nevals", "best", "bestError", "Eo", "env"]
    writeLog(mode=0, filename=filename, header=header)
    headerOPT = [f"opt{i}" for i in range(parameters["NPEAKS_MPB"])]
    writeLog(mode=0, filename=f"{path}/optima.csv", header=headerOPT)

    # Create the toolbox functions
    toolbox = createToolbox(parameters)

    # Check if the changes should be random or pre defined
    if(parameters["RANDOM_CHANGES"]):
        NCHANGES = len(parameters["NCHANGES"])
        changesEnv = [random.randint(parameters["RANGE_NEVALS_CHANGES"][0], parameters["RANGE_NEVALS_CHANGES"][1]) for _ in range(NCHANGES)]
    else:
        NCHANGES = len(parameters["CHANGES_NEVALS"])
        changesEnv = parameters["CHANGES_NEVALS"]

    # Main loop of ITER runs
    for run in range(1, RUNS+1):
        random.seed(run**5)
        best = None
        nevals = 0
        env = 1
        change = 0
        gen = 1
        genChangeEnv = 0
        flagEnv = 0
        eo_sum = 0

        # Initialize the benchmark for each run with seed being the minute
        rndBNK = random.Random()
        rndBNK.seed(int(seed)**5)
        if(parameters["BENCHMARK"] == "MPB"):
            fitFunction = movingpeaks.MovingPeaks(dim=parameters["NDIM"], random=rndBNK, **scenario)
        elif(parameters["BENCHMARK"] == "H1"):
            fitFunction = benchmarks.h1

        # Create the population with NSWARMS of size SWARMSIZE
        pop = [toolbox.swarm(n=SWARMSIZE) for _ in range(NSWARMS)]

        # Save the optima values
        if(peaks <= NCHANGES):
            saveOptima(parameters, fitFunction, path)
            peaks += 1

        # Repeat until reach the number of evals
        while nevals < NEVALS+1:

            # PSO
            for swarmId, swarm in enumerate(pop, 1):

                # Change detection
                if(parameters["CHANGE_DETECTION_OP"] and swarm.best):
                    change = changeDetection(swarm, toolbox, fitFunction, change=change, parameters=parameters)

                if(change and swarm):
                    swarm = reevaluateSwarm(swarm, toolbox, fitFunction, parameters=parameters)
                    best = None
                    if flagEnv == 0 :
                        env += 1
                        genChangeEnv = gen
                        flagEnv = 1

                for partId, part in enumerate(swarm, 1):

                    # Evaluate the particle
                    part.fitness.values = toolbox.evaluate(part, fitFunction, parameters=parameters)

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

                    # Save the log with all particles on it
                    if(parameters["LOG_ALL"]):
                        Eo = eo_sum / gen
                        log = [{"run": run, "gen": gen, "nevals":nevals, "swarmId": swarmId, "partId": partId, "part":part, "partError": part.best.fitness.values[0], "sbest": swarm.best, "sbestError": swarm.best.fitness.values[0], "best": best, "bestError": best.fitness.values[0], "Eo": Eo, "env": env}]
                        writeLog(mode=1, filename=filename, header=header, data=log)
                        # Debugation at particle level
                        if(parameters["DEBUG0"]):
                            print(log)

                for part in swarm:
                    toolbox.update(part, swarm.best)

            eo_sum += best.fitness.values[0]

            # Save the log only with the bests of each generation
            if(parameters["LOG_ALL"] == 0):
                Eo = eo_sum / gen
                log = [{"run": run, "gen": gen, "nevals":nevals, "best": best, "bestError": best.fitness.values[0], "Eo": Eo, "env": env}]
                writeLog(mode=1, filename=filename, header=header, data=log)

            # Debugation at generation level
            if(parameters["DEBUG1"]):
                Eo = eo_sum / gen
                print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}] Best:{best.fitness.values[0]:.4f}\tEo:{Eo:4.f}")

            if abs(gen - genChangeEnv) > 2:
                flagEnv = 0
                change = 0
            # End of the generation
            gen += 1

        # Debugation at generation level
        if(parameters["DEBUG2"]):
            Eo = eo_sum / gen
            print(f"[RUN:{run:02}][NGEN:{gen:04}][NEVALS:{nevals:06}] Eo:{Eo:.4f}")

    executionTime = (time.time() - startTime)
    if(parameters["DEBUG2"]):
        print(f"File generated: {path}/data.csv")
        print(f'Time Exec: {str(executionTime)} s\n')


    # Copy the config.ini file to the experiment dir
    if(parameters["CONFIG_COPY"]):
        shutil.copyfile("config.ini", f"{path}/config.ini")

    # Evaluate the offline error
    if(parameters["OFFLINE_ERROR"]):
        print("[METRICS]")
        os.system(f"python3 ../../metrics/offlineError.py -p {path}")

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

    # Read the parameters from the config file
    with open(f"{path}/config.ini") as f:
        parameters = json.loads(f.read())

    if path == ".":
        path = f"{parameters['PATH']}/{parameters['ALGORITHM']}"
        path = checkDirs(path)

    if(parameters["DEBUG2"]):
        print(f"======================================================")
        print(f"   AbCD Framework for Dynamic Optimization Problems")
        print(f"======================================================\n")
        print(f"[ALGORITHM SETUP]")
        print(f"- Name: {parameters['ALGORITHM']}")
        print()
        print(f"[BENCHMARK SETUP]")
        print(f"- Name: {parameters['BENCHMARK']}")
        print(f"- NDIM: {parameters['NDIM']}")


        print("\n[START]\n")

    # Call the algorithm
    pso(parameters, seed)
    print("\n[END]\nThx:)\n")

    # For automatic calling of the plot functions
    if(parameters["PLOT"]):
        os.system(f"python3 ../../PLOTcode.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")
        os.system(f"python3 ../../PLOT2code.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")


if __name__ == "__main__":
    main()


