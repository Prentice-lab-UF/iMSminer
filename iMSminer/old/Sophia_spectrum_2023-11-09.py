# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:06:43 2023

@author: yutinlin
"""

import time
import gc

#================Load MS Data=============#
import os
#os.chdir("Z:\\Lin\\2023\\Python\\Scripts")
os.chdir("\\\\10.247.46.200\\Prentice01\\Lin\\2023\\Python\\Scripts")
from IMS_Processing_Functions import get_ms_info, lwr_upr_mz, get_int, int_at_mz, normalize, SCiLS_raw_intensities


import pandas as pd
import numpy as np
from numpy import inf,nan
import pandarallel
import multiprocessing as mp
import numexpr as ne
from bokeh.plotting import figure, show
import seaborn as sns
import matplotlib.pyplot as plt

analyte = input("What is your analyte of interest? ") or "CL"
directory = input("What is your SCiLS dataset directory? ") or  "\\\\10.247.46.200\\Prentice01\\Lin\\2023\\SCiLS\\BTHS\\BTHS_CL"#directory = Z:\Lin\2023\SCiLS\BTHS\BTHS_CL
data_dir = input("Where to save the spectral data? ") or "\\\\10.247.46.200\\Prentice01\\Lin\\2023\\Python\\\Data\\BTHS\\CL_spectral_matrix"  #data_dir = Z:\Lin\2023\Python\Data\BTHS\CL_spectral_matrix


'''
analyte = input("What is your analyte of interest? ")
directory = input("What is your SCiLS dataset directory? ") #directory = Z:\Lin\2023\SCiLS\BTHS\BTHS_CL
data_dir = input("Where to save the spectral data? ") #data_dir = Z:\Lin\2023\Python\Data\BTHS\CL_spectral_matrix
'''

if not os.path.exists(data_dir):
        os.makedirs(data_dir)

if 'global_spectral_avg' in locals(): 
    del global_spectral_avg
    
nf = '157cal'
for file_name in os.listdir(directory):
    if file_name.endswith(".sbd"):       
        print(file_name)

        norI_dict = dict()
        ms_dict = get_ms_info(directory, file_name, nf)
        del ms_dict['coords'] 
        try:
            global_spectral_avg  # Attempt to access the variable
            # Object is defined
            pass
        except NameError:
            # Object is undefined
            global_spectral_avg = np.empty((len(ms_dict['mz']), 0), dtype=object)
        global_spectral_avg = np.insert(global_spectral_avg,0,np.mean(ms_dict["I"], axis = 0),axis=1)
        del ms_dict

global_spectral_avg_mean = np.mean(global_spectral_avg, axis=1)


## IGNORE ABOVE AS LONG AS IT WORKS


## USE YOUR NEW PEAK PICKING AND MASS ALIGNMENT ALGORITHM (SEE IF IT WORKS)

#===============Peak Picking====================#


file_name=os.listdir(directory)[0]

norI_dict = dict()
ms_dict = get_ms_info(directory, file_name, nf)


lwr = input("What is the lower bound? ")
upr = input("What is the upper bound? ")

noise = np.mean(ms_dict["I"][:,lwr & upr].std(axis=0))  #DO NOT USE I_mean. Watch low noise level, artifact of indexed row names.

select_I = pd.DataFrame({'mz': ms_dict['mz'], 'intensity': global_spectral_avg_mean}, columns=['mz', 'intensity']).set_index('mz') #set index to display mz instead of indices of intensity

######METHOD 2
from scipy.signal import find_peaks, peak_widths
from scipy import stats

select_I.reset_index(inplace=True)


h = 10*noise
dist = 10
p2, info = find_peaks(x=global_spectral_avg_mean,
height=h,
 distance=dist
 )

plt.plot(ms_dict['mz'],global_spectral_avg_mean)
plt.scatter(ms_dict['mz'][p2], global_spectral_avg_mean[p2], color='red', label='Data Points')

## LOCK MASS IS CALIBRATION. AND IGNORE FOR NOW

#===============lock mass====================================#
if analyte == "CL":
    LM_ref = 885.54930423
elif analyte == "FA":
    LM_ref = 885.54930423
elif analyte == "metabolite":
    LM_ref = 157.07657330
    
mz_885 = np.logical_and(ms_dict['mz'] >= (LM_ref-0.01), ms_dict['mz'] <= (LM_ref+0.01))

for i in range(ms_dict["I"].shape[0]):
    I_885 = ms_dict["I"][i][mz_885]
    I885_index = np.argmin(abs(ms_dict['mz']-LM_ref))
    lm_index = np.where(mz_885)[0][np.argsort(I_885)[::-1][0]]
    shift = I885_index-lm_index
    ms_dict['mz'] += (LM_ref-ms_dict['mz'][I885_index])
    print(shift)
    ms_dict["I"][i] = np.roll(ms_dict["I"][i], shift)
    if shift > 0:
        ms_dict["I"][i][-shift:]  = 0
#===========================================================#


############ THIS IS THE IMPORTANT PART ##################

## REPLACE PEAK PICKING AND MASS ALIGNMENT WITH YOUR CODES
## WRITE A MASS SPECTRUM FUNCTION 

if analyte == "CL":
    def get_mass_spectrum,
    plt.figure(figsize=(9, 3))
    plt.plot(ms_dict["mz"], global_spectral_avg_mean, color='red', alpha = 1, linewidth=0.5)
    plt.xlim(900, 1600) # GENERALIZE
    x_limits = plt.xlim()
    indices = np.where(np.logical_and(ms_dict["mz"] >= x_limits[0], ms_dict["mz"] <= x_limits[1]))[0]  # Access the first element of the tuple
    y_max = max(global_spectral_avg_mean)
    plt.ylim(0, y_max)
    plt.xlabel('m/z')
    plt.ylabel("Intensity [a.u.]")
    plt.subplots_adjust(bottom=0.2)  
   # plt.savefig("Z:\\Lin\\2023\\Python\\Data\\BTHS\\spectrum\\" + analyte + "_900-1600.jpg", dpi=1000)
    plt.show()
    #plt.close()