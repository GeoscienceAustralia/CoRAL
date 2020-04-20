"""
This python module contains functions for reading Synthetic Aperture 
Radar images
"""
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import sys, os


def readfile(file, sub_im, cr):
    """Function to read an image file"""

    root, ext = os.path.splitext(file)

    if ext == '.tif':
        print('Reading tiff image:', file)
        par = readpar(root + '.mli.par')
        data = readimg(file, sub_im, cr)

    elif ext == '.img':
        print('Reading ENVI image:', file)
        #par = readhdr(root + '.hdr')
        data = readimg(file, sub_im, cr)

    else: # must be GAMMA flat binary float format
        print('Reading flat binary image', file)
        par = readpar(root + ext + '.par')
        data = readmli(file, par, sub_im, cr)

    # extract relevant metadata
    rho_r = float(par['range_pixel_spacing'].split()[0])
    rho_a = float(par['azimuth_pixel_spacing'].split()[0])
    theta = float(par['incidence_angle'].split()[0])

    return data, rho_r, rho_a, theta


def readpar(file):
    """Function to read a GAMMA 'par' file into a dictionary"""
    par={}
    with open(file) as f:
        for line in f:
            if "Gamma" or " " in line:
                break # ignore header line
        for line in f:
            line=line.rstrip() # remove blank lines and whitespace
            if line and not "title" in line:
                (key, val) = line.split(":")
                par[str(key)] = val
    return par


def readmli(datafile, par, sub_im, cr):
    """Function to read a GAMMA mli file and provide a subsetted image"""
    ct = int(par['range_samples']) * int(par['azimuth_lines'])

    dt = np.dtype('>f4') # GAMMA files are big endian 32 bit float

    d = np.fromfile(datafile, dtype=dt, count=ct)

    d = d.reshape(int(par['azimuth_lines']), int(par['range_samples']))
    #print("Number of elements and size of the array is",d.size, d.shape)
    #d[d==0]= np.nan # convert zeros to nan
    return d[cr[1]-sub_im:cr[1]+sub_im,cr[0]-sub_im:cr[0]+sub_im]


def readimg(datafile, sub_im, cr):
    """Function to read a file and provide a subsetted image"""

    with rio.open(datafile) as src:
        d = src.read(1, window=Window(cr[0]-sub_im, cr[1]-sub_im, sub_im*2, sub_im*2))

    #print("Number of elements and size of the array is",d.size, d.shape)
    #d[d==0]= np.nan # convert zeros to nan
    return d
    

def read_radar_coords(filename):
    print("Reading textfile with CR positions...")
    site = []
    az = []
    rg = []
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        print("File", filename, "exists and is readable.")
        f = open(filename)
        lines = f.readlines()
        idx = 0
        for line in lines:
            # get site name
            line = line.strip("\n")
            site.append(line.split("\t")[0])
            az.append(line.split("\t")[-2])
            rg.append(line.split("\t")[-1])
            idx = idx + 1  
        print("Radar coordinates at %d sites read" % (idx))
        f.close()
    else:
        print("ERROR: Can't read file", filename)
        sys.exit()
    print()
    return site, az, rg
    
    
def write_radar_coords(filename_init, filename, sites, geom):
    print("Writing new CR positions to textfile...")
    fout = open(filename,'w')
    if os.path.isfile(filename_init) and os.access(filename_init, os.R_OK):
        print("File", filename_init, "exists and is readable.")
        f = open(filename_init)
        lines = f.readlines()
        for line in lines:
            # get site name
            name = line.split("\t")[0]
            cr = sites[name]
            temp = line.split("\t")[:-2]
            temp2 = "\t".join(temp)
            if geom == "asc":
                out_line = temp2 + "\t" + str(cr[0][1]) + \
                       "\t" + str(cr[0][0]) + "\n"
            if geom == "desc":
                out_line = temp2 + "\t" + str(cr[1][1]) + \
                       "\t" + str(cr[1][0]) + "\n"           
            fout.write(out_line)
        f.close()
        fout.close()    
    else:
        print("ERROR: Can't read file", filename_init)
        sys.exit()
    print()
    print("Azimuth and Range number of CR pixels has been written to " + \
    filename + ".")
    print()
    return
