# -*- coding: utf-8 -*-
"""
@ author: Jason Ang, Sophia Dalda, YutinLin
"""


import pandas as pd
import numpy as np
from numpy import inf,nan
import pandarallel
import multiprocessing as mp
import numexpr as ne
from bokeh.plotting import figure, show
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import inf,nan
import pandarallel
import multiprocessing as mp
import numexpr as ne
from bokeh.plotting import figure, show
import plotly.express as px
import os
import time
import gc
from IMS_Processing_Functions import get_ms_info, lwr_upr_mz, get_int, int_at_mz, normalize, SCiLS_raw_intensities


class VisualSpec:
    def __init__(self, nf, ms_dict, anayte, directory, data_dir):
        self.nf = nf
        self.ms_dict = ms_dict
        self.anayte = anayte
        self.directory = directory
        self.data_dir = data_dir
        self.global_spectral_avg_mean = self.global_spectral_avg_mean()
    
    def get_image_index(self, mz_low, mz_high):
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
        
        mz = self.ms_dict['mz']
        return np.where(np.logical_and(mz >= mz_low, mz <= mz_high))


    def get_image(self, output_file = "dynamic_plot.html"):
        '''
        Parameters=
        ----------
            MS data from get_ms_info

        Returns
        -------
        A heatmap
        '''
        
        x_max = self.ms_dict["coords"][:,0].max() 
        if self.ms_dict["coords"][:,1].min() < 0:
            self.ms_dict["coords"][:,1] -= min(self.ms_dict["coords"][:,1])
        y_max = self.ms_dict["coords"][:,1].max()
        
        # generate x,y rectangle and insert x,y pixels
        image_index = self.get_image_index(1447, 1457)
        image_mz = np.sum(self.ms_dict["I"][:,image_index].reshape(-1,len(image_index[0])), axis=1)
        image_matrix = np.zeros([x_max+1, y_max+1])
        i = 0
        for x, y in self.ms_dict["coords"]:
            image_matrix[x,y] = image_mz[i]
            i += 1
        
        #min/max of color bar
        min_intensity = np.min(self.ms_dict["I"])
        max_intensity = np.max(self.ms_dict["I"])
        
        
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
    def global_spectral_avg_mean(self):
        for file_name in os.listdir(self.directory):
            if file_name.endswith(".sbd"):       
                print(file_name)

                norI_dict = dict()
                self.ms_dict = get_ms_info(self.directory, file_name, self.nf)
                del self.ms_dict['coords'] 
                try:
                    global_spectral_avg  # Attempt to access the variable
                    # Object is defined
                    pass
                except NameError:
                    # Object is undefined
                    global_spectral_avg = np.empty((len(self.ms_dict['mz']), 0), dtype=object)
                global_spectral_avg = np.insert(global_spectral_avg,0,np.mean(self.ms_dict["I"], axis = 0),axis=1)
                del self.ms_dict

        global_spectral_avg_mean = np.mean(global_spectral_avg, axis=1)
        return global_spectral_avg_mean
    def peak_picking(self):
        #===============Peak Picking====================#


        file_name=os.listdir(self.directory)[0]

        norI_dict = dict()
        self.ms_dict = get_ms_info(self.directory, file_name, self.nf)


        if self.analyte == "CL":
            lwr, upr = self.ms_dict['mz'] >= 630, self.ms_dict['mz'] <= 650 #CL
        elif self.analyte == "FA":
            lwr, upr = self.ms_dict['mz'] >= 1660, self.ms_dict['mz'] <= 1700 #FA
        elif self.analyte == "metabolite":
            lwr, upr = self.ms_dict['mz'] >= 492, self.ms_dict['mz'] <= 495 #metabolite

        noise = np.mean(self.ms_dict["I"][:,lwr & upr].std(axis=0))  #DO NOT USE I_mean. Watch low noise level, artifact of indexed row names.

        select_I = pd.DataFrame({'mz': self.ms_dict['mz'], 'intensity': self.global_spectral_avg_mean}, columns=['mz', 'intensity']).set_index('mz') #set index to display mz instead of indices of intensity

        ######METHOD 2
        from scipy.signal import find_peaks, peak_widths
        from scipy import stats

        select_I.reset_index(inplace=True)


        h = 10*noise
        dist = 10
        p2, info = find_peaks(x=self.global_spectral_avg_mean,
        height=h,
         distance=dist
         )

        plt.plot(self.ms_dict['mz'],self.global_spectral_avg_mean)
        plt.scatter(self.ms_dict['mz'][p2], self.global_spectral_avg_mean[p2], color='red', label='Data Points')

    def lock_mass(self):
        ## LOCK MASS IS CALIBRATION. AND IGNORE FOR NOW

        #===============lock mass====================================#
        if self.analyte == "CL":
            LM_ref = 885.54930423
        elif self.analyte == "FA":
            LM_ref = 885.54930423
        elif self.analyte == "metabolite":
            LM_ref = 157.07657330
            
        mz_885 = np.logical_and(self.ms_dict['mz'] >= (LM_ref-0.01), self.ms_dict['mz'] <= (LM_ref+0.01))

        for i in range(self.ms_dict["I"].shape[0]):
            I_885 = self.ms_dict["I"][i][mz_885]
            I885_index = np.argmin(abs(self.ms_dict['mz']-LM_ref))
            lm_index = np.where(mz_885)[0][np.argsort(I_885)[::-1][0]]
            shift = I885_index-lm_index
            self.ms_dict['mz'] += (LM_ref-self.ms_dict['mz'][I885_index])
            print(shift)
            self.ms_dict["I"][i] = np.roll(self.ms_dict["I"][i], shift)
            if shift > 0:
                self.ms_dict["I"][i][-shift:]  = 0
        #===========================================================#
        
    def get_mass_spectrometer(self):
        ############ THIS IS THE IMPORTANT PART ##################

        ## REPLACE PEAK PICKING AND MASS ALIGNMENT WITH YOUR CODES
        ## WRITE A MASS SPECTRUM FUNCTION 

        if self.analyte == "CL":
            plt.figure(figsize=(9, 3))
            plt.plot(self.ms_dict["mz"], self.global_spectral_avg_mean, color='red', alpha = 1, linewidth=0.5)
            plt.xlim(900, 1600) # GENERALIZE
            x_limits = plt.xlim()
            indices = np.where(np.logical_and(self.ms_dict["mz"] >= x_limits[0], self.ms_dict["mz"] <= x_limits[1]))[0]  # Access the first element of the tuple
            y_max = max(self.global_spectral_avg_mean)
            plt.ylim(0, y_max)
            plt.xlabel('m/z')
            plt.ylabel("Intensity [a.u.]")
            plt.subplots_adjust(bottom=0.2)  
           # plt.savefig("Z:\\Lin\\2023\\Python\\Data\\BTHS\\spectrum\\" + analyte + "_900-1600.jpg", dpi=1000)
            plt.show()
            #plt.close()
            
def main():
    while True:
        analyte = input("What is your analyte of interest? ") or "CL"
        directory = input("What is your SCiLS dataset directory? ") or  "\\\\10.247.46.200\\Prentice01\\Lin\\2023\\SCiLS\\BTHS\\BTHS_CL"#directory = Z:\Lin\2023\SCiLS\BTHS\BTHS_CL
        data_dir = input("Where to save the spectral data? ") or "\\\\10.247.46.200\\Prentice01\\Lin\\2023\\Python\\\Data\\BTHS\\CL_spectral_matrix"  #data_dir = Z:\Lin\2023\Python\Data\BTHS\CL_spectral_matrix
        
        #================Load MS Data=============#
        os.chdir("\\\\10.247.46.200\\Prentice01\\Lin\\2023\\Python\\Scripts")
         
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        if 'global_spectral_avg' in locals(): 
            del global_spectral_avg
         
        nf = '157cal'
        file_name = os.listdir(directory)[0]
        
        norI_dict = dict()
        ms_dict = get_ms_info(directory, file_name, nf)
        #================Load MS Data=============#
        
        # Create object VisualSpec
        object_visualizer = VisualSpec(nf = nf, ms_dict = ms_dict, anayte = analyte, directory = directory, data_dir = data_dir)
         
         

if __name__ == '__main__':
    main()      