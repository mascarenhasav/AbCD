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
import getopt

cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

def writeTXT(data, name, path, std):
    if(std):
        line = f"{data[0]:.5f}\t{data[1]:.5f}"
    else:
        line = f"{data:.5f}"
    f = open(f"{path}/{name}.txt","w")
    f.write(line)
    f.close()

def offlineError(path, std=1):
    df = pd.read_csv(f"{path}/data.csv")

    luffy = df.drop_duplicates(subset=["run"], keep="last")[["Eo"]]
    eo = [np.mean(luffy["Eo"]), np.std(luffy["Eo"])]

    if(std):
        return eo
    else:
        return eo[0]


def main():
    startTime = time.time()
    arg_help = "{0} -p <path>".format(sys.argv[0])
    path = "."

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hp:", ["help", "path="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-p", "--path"):
            path = arg

    with open(f"{sys.path[0]}/config.ini") as f:
        parameters = json.loads(f.read())

    # Evaluate the offline error
    Eo = offlineError(path, std = parameters["STD_Eo"])
    writeTXT(Eo, "offlineError", path, std = parameters["STD_Eo"])
    if(parameters["DEBUG"]):
        if(parameters["STD_Eo"]):
            print(f"\n[Offline Error]: {Eo[0]:.5f}({Eo[1]:.5f})")
        else:
            print(f"\n[Offline Error]: {Eo:.5f}")

    executionTime = (time.time() - startTime)
    if(parameters["DEBUG"]):
        print(f"File generated: {path}/offlineError.txt")
        print(f'Time Exec: {str(executionTime)} s')


if __name__ == "__main__":
    main()
