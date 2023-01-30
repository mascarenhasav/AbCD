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
eo_sum = 0
best = None


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
Apply PSO on the particle
'''
def PSO_particle(part, best, parameters):
    X = (parameters["w"] for _ in range(len(part)))
    u1 = (random.uniform(0, parameters["phi1"]) for _ in range(len(part)))
    u2 = (random.uniform(0, parameters["phi2"]) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    #part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    part.speed = list(map(operator.mul, map(operator.add, part.speed, map(operator.add, v_u1, v_u2)), X))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))

    return part


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
    global eo_sum
    global best
    fitInd = function(x)[0]
    if(parameters["BENCHMARK"] == "MPB"):
        globalOP = function.maximums()[0][0]
    elif(parameters["BENCHMARK"] == "H1"):
        globalOP = function([8.6998, 6.7665])[0]

    fitness = [abs( fitInd - globalOP )]
    nevals += 1
    if best:
        eo_sum += best.fitness.values[0]

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
Anti-convergence operator
'''
def antiConvergence(pop, typeInd, parameters, randomInit):
    rconv = parameters["RCONV"]
    wswarmId = None
    wswarm = None
    nconv = 0
    for swarmId, swarm in enumerate(pop):
        # Compute the diameter of the swarm
        for p1Id, p1 in enumerate(swarm, 1):
            for p2Id, p2 in enumerate(swarm, 1):
                if (p1 == p2) or (typeInd[p1Id] != 1) or (typeInd[p2Id] != 1):
                    break
                for x1, x2 in zip(p1, p2):
                    d = math.sqrt( (x1 - x2)**2 ) # Euclidean distance between the components
                    if d >= rconv:  # If any is greater or equal rconv, not converged
                        nconv += 1
                        break
        # Search for the worst swarm according to its global best
        if not wswarm or swarm.best.fitness < wswarm.best.fitness:
            wswarmId = swarmId
            wswarm = swarm

    # If all swarms have converged, remember to randomize the worst
    if nconv == 0:
        randomInit[wswarmId] = 1

    return randomInit


'''
Exclusion operator
'''
def exclusion(pop, parameters, randomInit):
    rexcl = parameters["REXCL"]
    for s1, s2 in itertools.combinations(range(len(pop)), 2):
        # Swarms must have a best and not already be set to reinitialize
        if pop[s1].best and pop[s2].best and not (randomInit[s1] or randomInit[s2]):
            dist = 0
            for x1, x2 in zip(pop[s1].best, pop[s2].best):
                dist += (x1 - x2)**2
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
    rcloud = parameters["RCLOUD"]
    for i in range(parameters["NDIM"]):
        part[i] = sbest[i] + P*(random.uniform(-1, 1)*rcloud)
    return part


'''
Apply LS on the best
'''
def localSearch(best, toolbox, parameters, fitFunction):
    rls  = parameters["RLS"]
    bp = creator.Particle(best)
    for _ in range(parameters["ETRY"]):
        for i in range(parameters["NDIM"]):
            bp[i] = bp[i] + random.uniform(-1, 1)*rls
        bp.fitness.values = evaluate(bp, fitFunction, parameters=parameters)
        if bp.fitness > best.fitness:
            best = creator.Particle(bp)
            best.fitness.values = bp.fitness.values
    return best


'''
Check if a change occurred in the environment
'''
def changeDetection(swarm, toolbox, fitFunction, change, parameters):
    sensor = evaluate(swarm.best, fitFunction, parameters=parameters)
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
Update the best individuals
'''
def updateBest(part, swarm, best):
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

    return part, swarm, best


'''
Algorithm
'''
def abcd(parameters, seed):
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
    global best
    global eo_sum

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
        flagEnv = 0
        genChangeEnv = 0
        eo_sum = 0
        randomInit = [0 for _ in range(1, NSWARMS+2)]
        typeInd = [1 for i in range(1, SWARMSIZE+2)]
        if (parameters["RCLOUD"] >= 0) and (parameters["ES_PARTICLE_PERC"] > 0):
            for i in range(1, int(parameters["ES_PARTICLE_PERC"]*SWARMSIZE)+1):
                typeInd[i] = 2

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

            # Anti-convergence operator
            if(parameters["ANTI_CONVERGENCE_OP"] and parameters["RCONV"] > 0 and gen > 2):
                randomInit = antiConvergence(pop, typeInd, parameters, randomInit)

            # Exclusion operator
            if(parameters["EXCLUSION_OP"] and parameters["REXCL"] > 0 and gen > 2):
                randomInit = exclusion(pop, parameters, randomInit)

            # Local search operator
            if(parameters["LOCAL_SEARCH_OP"] and parameters["RLS"] > 0 and gen > 2):
                best = localSearch(best, toolbox, parameters, fitFunction)

            # PSO
            for swarmId, swarm in enumerate(pop, 1):

                # Change detection
                if(parameters["CHANGE_DETECTION_OP"] and swarm.best):
                    change = changeDetection(swarm, toolbox, fitFunction, change=change, parameters=parameters)
                    if(change and swarm):
                        swarm = reevaluateSwarm(swarm, toolbox, fitFunction, parameters=parameters)
                        best = None
                        if flagEnv == 0:
                            env += 1
                            genChangeEnv = gen
                            flagEnv = 1
                        randomInit[swarmId] = 0


                for partId, part in enumerate(swarm, 1):
                    if(gen > 2):
                        # If convergence or exclusion, randomize the particle
                        if(randomInit[swarmId]):
                            part = toolbox.particle()
                        else:
                            # Optimizer component
                            if(typeInd[partId] == 1):
                                part = PSO_particle(part, swarm.best, parameters)
                            elif(typeInd[partId] == 2):
                                part = ES_particle(part, swarm.best, parameters)


                    # Evaluates the individual
                    part.fitness.values = evaluate(part, fitFunction,  parameters=parameters)

                    # Updates the best
                    part, swarm, best = updateBest(part, swarm, best)

                    # Save the log with all particles on it
                    if(parameters["LOG_ALL"]):
                        Eo = eo_sum/nevals
                        log = [{"run": run, "gen": gen, "nevals":nevals, "swarmId": swarmId, "partId": partId, "part":part, "partError": part.best.fitness.values[0], "sbest": swarm.best, "sbestError": swarm.best.fitness.values[0], "best": best, "bestError": best.fitness.values[0], "Eo": Eo, "env": env}]
                        writeLog(mode=1, filename=filename, header=header, data=log)
                        # Debugation at particle level
                        if(parameters["DEBUG0"]):
                            print(log)

                # Randomization complete
                randomInit[swarmId] = 0

            # Save the log only with the bests of each generation
            if(parameters["LOG_ALL"] == 0):
                Eo = eo_sum/nevals
                log = [{"run": run, "gen": gen, "nevals":nevals, "best": best, "bestError": best.fitness.values[0], "Eo": Eo, "env": env}]
                writeLog(mode=1, filename=filename, header=header, data=log)

            # Debugation at generation level
            if(parameters["DEBUG1"]):
                Eo = eo_sum/nevals
                print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}] Best:{best.fitness.values[0]:.4f}\tEo:{Eo:.4f}")

            if abs(gen - genChangeEnv) > 2:
                flagEnv = 0
                change = 0
            # End of the generation
            gen += 1


        # Debugation at generation level
        if(parameters["DEBUG2"]):
            Eo = eo_sum/nevals
            print(f"[RUN:{run:02}][NGEN:{gen:04}][NEVALS:{nevals:06}] Eo: {Eo:.4f}")

    executionTime = (time.time() - startTime)
    if(parameters["DEBUG2"]):
        print(f"File generated: {path}/data.csv")
        print(f'Time Exec: {str(executionTime)} s\n')


    # Copy the config.ini file to the experiment dir
    if(parameters["CONFIG_COPY"]):
        shutil.copyfile("config.ini", f"{path}/config.ini")

    # Evaluate the offline error
    if(parameters["OFFLINE_ERROR"]):
        if (parameters["DEBUG2"]):
            print("[METRICS]")
            os.system(f"python3 {sys.path[0]}/metrics/offlineError.py -p {path} -d 1")
        else:
            os.system(f"python3 {sys.path[0]}/metrics/offlineError.py -p {path}")

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
        print(f"- Percentage of ind doing ES: {parameters['ES_PARTICLE_PERC']*100}%")
        if(parameters["ES_PARTICLE_PERC"] > 0):
            print(f"-- [ES]: Rcloud={parameters['RCLOUD']}")
        print(f"- Operators used:")
        if(parameters["EXCLUSION_OP"]):
            print(f"-- [Exlcusion]: Rexcl={parameters['REXCL']}")
        if(parameters["ANTI_CONVERGENCE_OP"]):
            print(f"-- [ANTI-CONVERGENCE]: Rconv={parameters['RCONV']}")
        if(parameters["LOCAL_SEARCH_OP"]):
            print(f"-- [LOCAL_SEARCH]: Rls={parameters['RLS']}")

        print()
        print(f"[BENCHMARK SETUP]")
        print(f"- Name: {parameters['BENCHMARK']}")
        print(f"- NDIM: {parameters['NDIM']}")

        print("\n[START]\n")

    # Call the algorithm
    abcd(parameters, seed)
    if (parameters["DEBUG2"]):
        print("\n[END]\nThx :)\n")

    # For automatic calling of the plot functions
    if(parameters["PLOT"]):
        os.system(f"python3 ../../PLOTcode.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")
        os.system(f"python3 ../../PLOT2code.py {parameters['ALGORITHM']} {year}-{month}-{day} {hour}-{minute}")


if __name__ == "__main__":
    main()


