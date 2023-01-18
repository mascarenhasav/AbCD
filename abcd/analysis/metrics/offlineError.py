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
import time

cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

def writeTXT(data, name, path):
    line = f"{data[0]:.5f}\t{data[1]:.5f}"
    #print(line)
    f = open(f"{path}/{name}.txt","w")
    f.write(line)
    f.close()

def offlineError(path, std=1):
    startTime = time.time()
    df = pd.read_csv(f"{path}/data.csv")
    data = [[] for i in range( len(pd.unique(df["run"])) )]
    for i in range(len(pd.unique(df["run"])) ): # Get the number of runs
        data[i] = df[df["run"] == i+1]
        data[i] = data[i].drop_duplicates(subset=["gen"], keep="last")[["gen", "nevals", "swarmId", "bestError", "env"]]
        data[i].reset_index(inplace=True)
        del data[i]["index"]
        #data[i].to_csv(f"{path}/run{i+1}.csv", index = True)

    NGEN = len(data[0]["gen"])
    NRUNS = len(data)
    luffy = [0 for _ in range(NRUNS)]

    for j in range(NRUNS):
        luffy[j] = np.sum(data[j]['bestError'])/NGEN

    eo = [np.mean(luffy), np.std(luffy)]

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    if(std):
        return eo
    else:
        return eo[0]


def main():
    # reading the parameters from the config file
    try:
        path = sys.argv[1]
    except IndexError:
        with open("./config.ini") as f:
            parameters = json.loads(f.read())
        debug = parameters["DEBUG"]
        if(debug):
            print("Parameters:")
            print(parameters)
        #path = f"{parameters['PATH']}/{parameters['ALGORITHM']}/{sys.argv[1]}/{sys.argv[2]}"

    Eo = offlineError(path)
    writeTXT(Eo, "offlineError", path)
    print(Eo)


if __name__ == "__main__":
    main()
