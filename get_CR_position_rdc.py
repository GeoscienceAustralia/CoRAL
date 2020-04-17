"""
Main script
- converts lat/lon coordinates of corner reflectors into azimuth and range
  pixel positions
- can be run after SLC images have been coregistered to a DEM and
  lat/lon coordinates have been derived (using calc_lat_lon_TxxxX)

INPUT:
txtfile: CR_site_lon_lat_hgt_date.txt located in ../GAMMA/TxxxX/
         *sar_latlon.txt file containing lon/lat values, located in ../GAMMA/TxxxX/DEM
         ./DEM/diff*.par file containing length/width of radar-coded DEM, located in ../GAMMA/TxxxX/DEM

OUTPUT:
textfile: CR_site_lon_lat_hgt_date_az_rg.txt
          same as Input file with two columns added containing
          azimuth and range position of central pixel at CR location

@author: Thomas Fuhrmann @ GA June, 2018

usage:
this script is currently located and executed in the TxxxX directory
Load python3 module on NCI: module load python3
then execute, e.g.: python3 get_CR_position_rdc.py
for effective parallel processing and if memory is exceeded, start an interactive session:
qsub -I -q express -lstorage=gdata/dg9,walltime=1:00:00,mem=32Gb,ncpus=7,wd
"""


# import modules
import sys
import os
import os.path
import math
import fnmatch
import numpy as np
# for parallel processing: module joblib has to be installed
from joblib import Parallel, delayed
import multiprocessing


#######################
### Path and file names
#######################
# give general processing path here
io_path = "/g/data/dg9/INSAR_ANALYSIS/CAMDEN/S1/GAMMA/T147D/"
# note that the GAMMA "DEM" folder is assumed inside this path
dem_path = io_path + "DEM/"
# give name of CR input file consisting of site name, longitude, latitude (+ optional column fields) 
cr_filename = "CR_site_lon_lat_hgt_date1_date2.txt"
# name of CR output file (automatic naming):
cr_filename_out = cr_filename[:-4] + "_az_rg_initial.txt"
########
# set number of processors here:
nCPU = 7
######## 



# Welome
print()
print("########################################")
print("# Get CR position in radar-coordinates #")
print("########################################")

print()


# Read CR_site_lon_lat_hgt_date1_date2.txt
print("Reading textfile with CR positions (latitude longitude height date)\
...")
filename_in = io_path + cr_filename
if os.path.isfile(filename_in) and os.access(filename_in, os.R_OK):
    print("File", filename_in, "exists and is readable.")
    f = open(filename_in)
    sites = f.readlines()
    sites[:] = [line.split("\t")[0] for line in sites]
    f = open(filename_in)
    lons = f.readlines()
    lons[:] = [float(line.split("\t")[1]) for line in lons]
    f = open(filename_in)
    lats = f.readlines()
    lats[:] = [float(line.split("\t")[2]) for line in lats]
else:
    print("ERROR: Can't read file", filename)
    sys.exit()
print()


# Read diff.par file
for file in os.listdir(dem_path):
    if fnmatch.fnmatch(file, "diff*.par"):
        filename = dem_path + file
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            f = open(filename)
            lines = f.readlines()
            for line in lines:
                if "range_samp_1" in line:
                    width = int(line.split()[1])
                if "az_samp_1" in line:
                    length = int(line.split()[1])
        else:
            print("ERROR: Can't read file", filename)
            sys.exit()
print("Width of radar-coded data (i.e. number of range samples):", width)
print("Length of radar-coded data (i.e. number of azimuth lines):", length)
print()


# Read lat/lon coordinate file
for file in os.listdir(dem_path):
    if fnmatch.fnmatch(file, "*sar_latlon.txt"):
        filename = dem_path + file
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            print("File", filename, "exists and is readable.")
        else:
            print("ERROR: Can't read file", filename)
            sys.exit()
print()


# Convert lat/lon diff to metric using the CR lat/lon values
# Constants
a = 6378137
f = 0.0033528107
e2 = 1/(1-f)**2-1
c = a*math.sqrt(1+e2)


def _inner(cr):
    lonCR = cr[0][0]
    latCR = cr[0][1]
    f = open(filename)
    lines = f.readlines()
    dist_sq_min = 999 # in deg
    az = 0
    rg = 0
    for line in lines:
        lat = float(line.split(" ")[0])
        lon = float(line.split(" ")[1])
        dist_sq = (lon-lonCR)**2 + (lat-latCR)**2
        if dist_sq < dist_sq_min:
            dist_sq_min = dist_sq
            az_CR = az
            rg_CR = rg
        if rg < width-1:
            rg = rg + 1
        else:
            rg = 0
            az = az + 1
    # save az and rg coordinates to list
    #az_CRs.append(az_CR)
    #rg_CRs.append(rg_CR)
    # print distance of pixel coordinate to CR coordinate
    dist_min = math.sqrt(dist_sq_min)
    V = math.sqrt((1+(e2*math.cos(latCR/180*math.pi)**2)))
    N = c/V
    M = c/V**3
    dist_min_m = dist_min/180*math.pi*math.sqrt(M*N)
    print("Pixel with minimum distance to CR location is:")
    print("azimuth: %d, range: %d" % (az_CR, rg_CR))
    print("Distance is %8.6f degree or %3.1f m" % (dist_min, dist_min_m))
    return az_CR, rg_CR 
print()


##################################
# function for parallel loop processing
keys = sites
values = (np.array([[lons[i], lats[i]]], dtype=float) \
             for i in range(0,len(sites)))
sites = dict(zip(keys, values))
names = sites.keys()
# note that dictionaries are unsorted and the variable name is hence not ordered
def processInput(name):
    cr = sites[name]
    az, rg = _inner(cr)
    return az, rg

# parallel processing of MLI read and RCS, SCR calculation
num_cores = multiprocessing.cpu_count()
# num_cores results in 32 on the NCI which in turn results in an error
# hence the number of 16 cores is hard-coded here
results = Parallel(n_jobs=nCPU)(delayed(processInput)(name) for name in names)
az_CRs = []
rg_CRs = []
for i in range(0,len(names)):
    az_CRs.append(results[i][0])
    rg_CRs.append(results[i][1])	
# print(az_CRs)
# print(rg_CRs)

# save the range and azimuth numbers of pixels in a new txt file
print("Reading textfile with CR positions (latitude longitude height date)\
...")
filename_out = io_path + cr_filename_out
fout = open(filename_out,'w')
if os.path.isfile(filename_in) and os.access(filename_in, os.R_OK):
    print("File", filename_in, "exists and is readable.")
    f = open(filename_in)
    lines = f.readlines()
    idx = 0
    for line in lines:
        out_line = line.strip("\n") + "\t" + str(az_CRs[idx]) + "\t" + \
        str(rg_CRs[idx]) + "\n"
        fout.write(out_line)
        idx = idx + 1
else:
    print("ERROR: Can't read file", filename_in)
    sys.exit()
fout.close()
print()
print("Azimuth and Range number of CR pixels has been written to " + \
filename_out + ".")
print()

