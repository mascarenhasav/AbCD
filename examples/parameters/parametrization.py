import numpy as np
import os
import json

config = {
"ALGORITHM": "ADPSO",
"BENCHMARK": "MPB",
"RUNS": 50,
"NEVALS": 100000,
"POPSIZE": 200,
"phi1": 0.729843788,
"phi2": 2.05,
"BOUNDS_POS": [0, 100],
"BOUNDS_VEL": [-5, 5],
"CHANGE_DETECTION_OP": 1,
"NSWARMS": 10,
"ES_PARTICLE_PERC": 1,
"ES_CHANGE_OP": 0,
"RCLOUD": 0,
"LOCAL_SEARCH_OP": 0,
"ETRY": 20,
"RLS": 0,
"EXCLUSION_OP": 0,
"REXCL": 0,
"ANTI_CONVERGENCE_OP": 1,
"RCONV": 0,
"NDIM": 10,
"CHANGE": 1,
"RANDOM_CHANGES": 0,
"RANGE_NEVALS_CHANGES": [100, 100000],
"NCHANGES": 19,
"CHANGES_NEVALS": [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000],
"NPEAKS_MPB": 10,
"PERIOD_MPB": 0,
"UNIFORM_HEIGHT_MPB": 0,
"MOVE_SEVERITY_MPB": 1,
"MIN_HEIGHT_MPB": 30,
"MAX_HEIGHT_MPB": 70,
"MIN_COORD_MPB": 0,
"MAX_COORD_MPB": 100,
"PATH": "../../../experiments",
"FILENAME": "data.csv",
"PLOT" : 0,
"DEBUG": 0
}

algorithm = "ADPSO-ES-1"
if(os.path.isdir(algorithm) == False):
    os.mkdir(algorithm)
parameter = "RCONV"

path = f"{algorithm}/{parameter}"
pathParameter = ""
if(os.path.isdir(path) == False):
    os.mkdir(path)

values = [round(i,2) for i in np.arange(3.0, 50.5, 0.5)]
#values = [0.0]

for i in values:
    config[parameter] = i
    pathParameter = path + f"/{i}"
    if(os.path.isdir(pathParameter) == False):
        os.mkdir(pathParameter)
    with open(f"{pathParameter}/config.ini", "w") as convert_file:
        convert_file.write(json.dumps(config))
    print(f"{pathParameter}")
    os.system(f"python3 ../../codes/algorithms/adpso/ADPSO.py -s 42 -p {pathParameter}")
    os.system(f"python3 ../../codes/analysis/offlineError.py {pathParameter} &")
