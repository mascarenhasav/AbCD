import numpy as np
import os
import json

config = {
"ALGORITHM": "mQSO",
"BENCHMARK": "MPB",
"RUNS": 50,
"NEVALS": 500000,
"POPSIZE": 100,
"phi1": 1.4962,
"phi2": 1.4962,
"w": 1,
"BOUNDS_POS": [0, 100],
"BOUNDS_VEL": [-5, 5],
"CHANGE_DETECTION_OP": 1,
"NSWARMS": 10,
"ES_PARTICLE_PERC": 0.5,
"ES_CHANGE_OP": 0,
"RCLOUD": 1,
"LOCAL_SEARCH_OP": 0,
"ETRY": 2,
"RLS": 2.35,
"EXCLUSION_OP": 1,
"REXCL": 22.9,
"ANTI_CONVERGENCE_OP": 1,
"RCONV": 39.7,
"CHANGE": 1,
"RANDOM_CHANGES": 0,
"NCHANGES": 99,
"RANGE_NEVALS_CHANGES": [5000, 10000, 30000, 70000, 72000],
"CHANGES_NEVALS": [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000, 125000, 130000, 135000, 140000, 145000, 150000, 155000, 160000, 165000, 170000, 175000, 180000, 185000, 190000, 195000, 200000, 205000, 210000, 215000, 220000, 225000, 230000, 235000, 240000, 245000, 250000, 255000, 260000, 265000, 270000, 275000, 280000, 285000, 290000, 295000, 300000, 305000, 310000, 315000, 320000, 325000, 330000, 335000, 340000, 345000, 350000, 355000, 360000, 365000, 370000, 375000, 380000, 385000, 390000, 395000, 400000, 405000, 410000, 415000, 420000, 425000, 430000, 435000, 440000, 445000, 450000, 455000, 460000, 465000, 470000, 475000, 480000, 485000, 490000, 495000],
"NDIM": 10,
"SCENARIO_MPB": 2,
"NPEAKS_MPB": 10,
"UNIFORM_HEIGHT_MPB": 50,
"MOVE_SEVERITY_MPB": 1,
"MIN_HEIGHT_MPB": 30,
"MAX_HEIGHT_MPB": 70,
"MIN_WIDTH_MPB": 1,
"MAX_WIDTH_MPB": 12,
"MIN_COORD_MPB": 0,
"MAX_COORD_MPB": 100,
"LAMBDA_MPB": 0,
"PATH": "../examples",
"FILENAME": "data.csv",
"LOG_ALL": 0,
"PLOT" : 0,
"CONFIG_COPY": 0,
"OFFLINE_ERROR": 1,
"BEBC_ERROR": 1,
"DEBUG2": 1,
"DEBUG1": 0,
"DEBUG0": 0
}

algorithm = "mQSO"
if(os.path.isdir(algorithm) == False):
    os.mkdir(algorithm)
parameter = "10-10-1"

path = f"{algorithm}/{parameter}"
pathParameter = ""
if(os.path.isdir(path) == False):
    os.mkdir(path)

#values = [round(i,2) for i in np.arange(0.5, 30, 0.5)]
#values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
#values = [10.0, 20.0, 30.0, 40.0, 50.0]
values = [100]

for i in values:
    #config[parameter] = i
    pathParameter = path + f"/{i}"
    if(os.path.isdir(pathParameter) == False):
        os.mkdir(pathParameter)
    with open(f"{pathParameter}/config.ini", "w") as convert_file:
        convert_file.write(json.dumps(config))
    print(f"{pathParameter}")
    os.system(f"python3 ../../abcd/abcd.py -s 412 -p {pathParameter}")
