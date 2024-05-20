# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:39:07 2024

@author: yutin
"""

#==============LOAD iMSminer FUNCTIONS================#

import os
os.chdir("/home/yutinlin/workspace/iMSminer")
from data_preprocessing import Preprocess
os.chdir("/home/yutinlin/workspace/iMSminer")
from data_analysis import DataAnalysis
import assorted_functions



#===========PREPROCESSING imzML==============#

preprocess = Preprocess()
preprocess.peak_pick(percent_RAM=5, method="point", generate_spectrum=True)
preprocess.run(percent_RAM=1, peak_alignment=False, align_halfwidth=2, 
               grid_iter_num=10, align_reduce=True, reduce_halfwidth=20, 
               plot_aligned_peak = True, index_peak_plot = 20, plot_num_peaks=5)






#===========ANALYZING PREPROCESSED imzML==============#
data_analysis = DataAnalysis()
data_analysis.load_preprocessed_data()
data_analysis.normalize_pixel(normalization="TIC")
data_analysis.calibrate_mz()
data_analysis.MS1_search()
data_analysis.filter_analytes()
data_analysis.image_clustering(k=10, perplexity=6, zoom=0.3, quantile=95)
data_analysis.insitu_clustering(k=4,perplexity=25, show_ROI=True, show_square=True)
data_analysis.make_FC_plot()
data_analysis.make_boxplot()
data_analysis.get_ion_image(replicate=0, show_ROI=True, show_square=True, color_scheme="inferno")

#p6_10 p6_35 pc_10 pc_35 gf_10 gf_35 cmc_10 cmc_35


