# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:38:20 2023

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
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

analyte = input("What is your analyte of interest? ")
directory = input("What is your SCiLS dataset directory? ") or  "\\\\10.247.46.200\\Prentice01\\Lin\\2023\\SCiLS\\BTHS\\BTHS_CL"#directory = Z:\Lin\2023\SCiLS\BTHS\BTHS_CL
data_dir = input("Where to save the spectral data? ") or "\\\\10.247.46.200\\Prentice01\\Lin\\2023\\Python\\\Data\\BTHS\\CL_spectral_matrix"  #data_dir = Z:\Lin\2023\Python\Data\BTHS\CL_spectral_matrix


if not os.path.exists(data_dir):
        os.makedirs(data_dir)

if 'global_spectral_avg' in locals(): 
    del global_spectral_avg
    
nf = '157cal'
file_name = os.listdir(directory)[0]

norI_dict = dict()
ms_dict = get_ms_info(directory, file_name, nf)


### WE DONT CARE ABOUT THE ABOVE AS LONG AS IT RUNS ##


## BELOW, WE ARE MAKING IMAGES

def get_image_index(mz_low, mz_high):
    '''
    Parameters
    ----------
    mz_low : a float or int
        lower m/z cutoff for image
    mz_high : a float or int
        higher m/z cutoff for image
        
    Returns
    -------
    indices corresponding to m/z range 
    '''
    
    mz = ms_dict['mz']
    return np.where(np.logical_and(mz >= mz_low, mz <= mz_high))


def get_image(ms_dict, output_file = "dynamic_plot.html"):
    '''
    Parameters=
    ----------
    ms_dict : dictionary 
        MS data from get_ms_info

    Returns
    -------
    A heatmap
    '''
    
    x_max = ms_dict["coords"][:,0].max() 
    if ms_dict["coords"][:,1].min() < 0:
        ms_dict["coords"][:,1] -= min(ms_dict["coords"][:,1])
    y_max = ms_dict["coords"][:,1].max()
    
    # generate x,y rectangle and insert x,y pixels
    image_index = get_image_index(1447, 1457)
    image_mz = np.sum(ms_dict["I"][:,image_index].reshape(-1,len(image_index[0])), axis=1)
    image_matrix = np.zeros([x_max+1, y_max+1])
    i = 0
    for x, y in ms_dict["coords"]:
        image_matrix[x,y] = image_mz[i]
        i += 1
    
    #min/max of color bar
    min_intensity = np.min(ms_dict["I"])
    max_intensity = np.max(ms_dict["I"])
    
    
    # Plotly Express heatmap
    fig = px.imshow(image_matrix.T, zmin=min_intensity, zmax=max_intensity, color_continuous_scale='Viridis')
    fig.update_layout(coloraxis_colorbar=dict(ticks='outside',
    tickvals=[min_intensity, max_intensity],
    ticktext=['{:.0%}'.format(min_intensity / max_intensity),
              '{:.0%}'.format(max_intensity / max_intensity)],
    len=0.4,
    y=0.5,
    yanchor='middle'),
    xaxis=dict(
       showticklabels=False, 
   ),
   yaxis=dict(
       showticklabels=False, 
  

   ))
    
    fig.update_layout(dragmode="lasso")

    fig.show()
    # Save the plot as an HTML file
    fig.write_html(output_file)

    # Store html for front end
    plot_html = fig.to_html()

    
    '''
    # plot image
    plt.imshow(image_matrix.T, vmin=min_intensity, vmax=max_intensity,)
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    # color bar customization
    colorbar = plt.colorbar(location ="right",shrink = 0.45)
    colorbar.set_ticks([min_intensity,max_intensity])
    colorbar.set_ticklabels(['{:.0%}'.format(min_intensity/max_intensity), '{:.0%}'.format(max_intensity/max_intensity)])
    plt.show
    '''    
    


    
    
    
    # find functions for the following
    # make color bar nicer (hints: min, max) [by max, e.g., np.max(ms_dict["I"])]
    # make x,y axes "nicer" (should we have numbers? etc. etc.)
    # hint: use the plt functions
    
    
    
get_image(ms_dict)
