'''
Code to plot graph using data file

Alexandre Mascarenhas
'''
import json
import shutil
import itertools
import operator
import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
import os
import csv
import sys

cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute


def mean(data):
    bMean = [0 for i in range( len(data[0]["bestError"]) )]
    bStd = [0 for i in range( len(data[0]["bestError"]) )]
    std = [0 for i in range( len(data) )]
    sum = 0

    for i in range (len(data[0]["bestError"])):
        for j in range(len(data)):
            sum += data[j]["bestError"][i]
            std[j] = data[j]["bestError"][i]
        bMean[i] = sum/len(data)
        bStd[i] = np.std(std)
        sum = 0
        std = [0 for i in range( len(data) )]

    zipped = list(zip(data[0]["gen"], bMean, bStd))
    bestMean = pd.DataFrame(zipped, columns=["gen", "bestError", "std"])
    return bestMean



def bestOfGeneration(data, changes, path):
    NGEN = len(data[0]["gen"])
    NRUNS = len(data)
    begRun = [0 for i in range (NRUNS)]
    begGen = [0 for i in range (NGEN+1)]

    for g in data[0]["gen"]:

        for j in range (len(data)):
            begRun[j] = data[j]['bestError'][g-1]

        begGen[g] = np.mean(begRun)

    Fbog = [np.mean(begGen), np.std(begGen)]
    return Fbog



def modifiedOfflineError(data, changes, path):
    NGEN = len(data[0]["gen"])
    NRUNS = len(data)
    NCHANGES = len(changes)
    emoChanges = [0 for _ in range(NCHANGES)]
    emoRun = [0 for _ in range(NRUNS)]

    for j in range(NRUNS):

        for i, n in zip(changes, range(NCHANGES)):
            emoChanges[n] = data[j]['bestError'][i-1]

        emoRun[j] = np.mean(emoChanges)

    Emo = [np.mean(emoRun), np.std(emoRun)]
    return Emo


def bestErrorBeforeChange(data, changes, path):
    NGEN = len(data[0]["gen"])
    NRUNS = len(data)
    NCHANGES = len(changes)
    bebcList = [0 for i in range (NCHANGES)]
    bebcRun = [0 for i in range (NRUNS)]

    for dataRun, r in zip(data, range(NRUNS)):

        for j, i in zip(changes, range(NCHANGES)):
            bebcList[i] = dataRun["bestError"][j-1]

        bebcRun[r] = np.mean(bebcList)

    Eb = [np.mean(bebcRun), np.std(bebcRun)]
    return Eb


def writeTXT(data, name, path):
    line = f"- {name}= {data[0]:.4f}({data[1]:.4f})\n"
    print(line)
    f = open(f"{path}/results.txt","a")
    f.write(line)
    f.close()


def main():
    # reading the parameters from the config file
    with open("./config.ini") as f:
        parameters = json.loads(f.read())
    debug = parameters["DEBUG"]
    if(debug):
        print("Parameters:")
        print(parameters)

    path = f"{parameters['PATH']}/{parameters['ALGORITHM']}/{sys.argv[1]}/{sys.argv[2]}"
    df = pd.read_csv(f"{path}/data.csv")

    Fbog = [-1, -1]
    Emo = [-1, -1]
    Eb = [-1, -1]

    path2 = f"{path}/subsets"
    if(os.path.isdir(path2) == False):
        os.mkdir(path2)

    # Separate the dataset em subsets filtered by runs
    data = [[] for i in range( len(pd.unique(df["run"])) )]
    swarm = [[] for i in range( len(pd.unique(df["swarmId"])) )]
    for i in range(len(pd.unique(df["run"])) ): # Get the number of runs
        data[i] = df[df["run"] == i+1]
        for j in range(len(pd.unique(data[i]["swarmId"])) ): # Get the number of gens
            swarm[j] = data[i][data[i]["swarmId"] == j+1]
            swarm[j] = swarm[j].drop_duplicates(subset=["gen"], keep="last")[["gen", "nevals", "swarmId", "sbestError", "env"]]
            swarm[j].reset_index(inplace=True)
            del swarm[j]["index"]
            swarm[j].to_csv(f"{path}/subsets/run{i+1}-swarm{j+1}.csv", index = True)
        data[i] = data[i].drop_duplicates(subset=["gen"], keep="last")[["gen", "nevals", "swarmId", "bestError", "env"]]
        data[i].reset_index(inplace=True)
        del data[i]["index"]
        data[i].to_csv(f"{path}/subsets/run{i+1}.csv", index = True)



    # Get the points of changes of environment
    changesEnv = data[0].ne(data[0].shift()).filter(like="env").apply(lambda x: x.index[x].tolist())["env"][1:]
    changes = [0 for _ in range(len(changesEnv))]
    for i, n in zip(changesEnv, range(len(changesEnv))):
        changes[n] = int(i)

    header = ["algorith", "benchmark", "fbog", "fbog_sd", "emo", "emo_sd", "eb", "eb_sd"]
    with open(f"{path}/results.csv", "w") as file:
        csvwriter = csv.DictWriter(file, fieldnames=header)
        csvwriter.writeheader()

    f = open(f"{path}/results.txt","w")
    lines = [\
    "-----------------------------------------------------\n",\
    "              Summary of the Experiment              \n\n",\
    f"Experiment: {parameters['ALGORITHM']} on {parameters['BENCHMARK']}\n\n",\
    "To know the setup of experiment, please see\n",\
    "the file 'config.ini' in this same directory.  \n\n",\
    "Results:\n\n",\
    ]
    f.writelines(lines)
    f.close()

    # Print
    print(f"------------------------------")
    print(f"       Analysis results      ")
    print()
    if(parameters["FBOG"]):
        Fbog = bestOfGeneration(data, changes, path)
        writeTXT(Fbog, "Fbog", path)
    if(parameters["EMO"]):
        Emo = modifiedOfflineError(data, changes, path)
        writeTXT(Emo, "Emo ", path)
    if(parameters["EB"]):
        Eb = bestErrorBeforeChange(data, changes, path)
        writeTXT(Eb, "Eb  ", path)
    print(f"------------------------------")


    # CSV file
    line = [{"algorith":parameters["ALGORITHM"], \
    "benchmark":parameters["BENCHMARK"], \
    "fbog":Fbog[0], \
    "fbog_sd":Fbog[1], \
    "emo":Emo[0], \
    "emo_sd":Emo[1], \
    "eb":Eb[0], \
    "eb_sd":Eb[1]
    }]
    with open(f"{path}/results.csv", mode="a") as file:
        csvwriter = csv.DictWriter(file, fieldnames=header)
        csvwriter.writerows(line)


    f = open(f"{path}/results.txt","a")
    lines = [\
    f"\n\n\nScript executed in: {year}/{month}/{day} {hour}:{minute}\n",\
    "\n                       Is us!                      \n",\
    "-----------------------------------------------------\n",\
    ]
    f.writelines(lines)
    f.close()

    if(parameters["DELETE_RAW_DATA"]):
        os.system(f"rm {path}/data.csv")


if __name__ == "__main__":
    main()
