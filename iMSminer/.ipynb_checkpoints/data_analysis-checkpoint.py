# -*- coding: utf-8 -*-
"""
Created on Thu May  2 02:22:26 2024

@author: yutin
"""

import gc
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from itertools import permutations, combinations
import subprocess
import seaborn as sns
import math
from statannotations.Annotator import Annotator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# DataAnalysis class
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec


os.chdir("/home/yutinlin/workspace/iMSminer")
from assorted_functions import make_image, make_image_1c, make_image_1c_njit, max_normalize, clustering_in_embedding, plot_image_at_point, gromov_trick, significance, draw_ROI,bbox_select



class DataAnalysis:
    
    def __init__(
            self
    ):
        
        detect_gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if detect_gpu.returncode == 0:
            print("GPU Detected:\n", detect_gpu.stdout.strip())
            
            test_library = ['cupy', 'cudf', 'cuml']
            for lib in test_library:
                try:
                    __import__(lib)
                    print(f"{lib.capitalize()} is installed and imported successfully!")
                    self.gpu = True  
                except ImportError:
                    print(f"{lib.capitalize()} is not installed or could not be imported.")
                    self.gpu = False
                    break     
        else:
            print("No GPU detected or NVIDIA driver not installed.")
            self.gpu = False
            
        try:
            __import__("numba")
            print(f"Numba is installed and imported successfully!")
            self.jit = True 
        except:
            print(f"Numba is not installed or could not be imported.")
            self.jit = False
            
        self.data_dir = input("What is your image dataset directory? ") 
        self.ROI_num = int(input("How many ROIs are you analyzing? "))
        
        fig_ratio_chosen = False 
        while not fig_ratio_chosen:
            fig_ratio = input("Render figures with `small`, `medium`, or `large` ratio? ")
            if fig_ratio == "small":
                self.fig_ratio = 10
                fig_ratio_chosen = True
            elif fig_ratio == "medium":
                self.fig_ratio = 12.5
                fig_ratio_chosen = True
            elif fig_ratio == "large":
                self.fig_ratio = 15
                fig_ratio_chosen = True
            else:
                fig_ratio_chosen = False
                print("Select one of the options.")
            
        
    def load_preprocessed_data(self):
        datasets = os.listdir(self.data_dir)
        datasets = np.asarray(datasets)[(np.char.find(datasets, "coords") == -1) & (np.char.find(datasets, "csv") != -1)]


        df_mean_all = pd.DataFrame()
        self.df_pixel_all = pd.DataFrame()
        ROI_edge = pd.DataFrame()
            
        for rep, dataset in enumerate(datasets):
            print(f"Importing {dataset}.")
            df_build = pd.read_csv(f"{self.data_dir}/{dataset}")
            print(f"Finished importing {dataset}.")
            
            df_build.rename(columns={'Unnamed: 0': 'mz'}, inplace=True)
            df_build = df_build.T
            
            self.mz = df_build.loc['mz']
            df_build.drop('mz', inplace=True)
            
            df_coords = pd.read_csv(f"{self.data_dir}/{dataset[:-4]}_coords.csv")
            df_coords.rename(columns={'0': 'x', '1': 'y'}, inplace=True)
            df_coords['x'] = df_coords['x'] - np.min(df_coords['x']) + 1
            df_coords['y'] = df_coords['y'] - np.min(df_coords['y']) + 1
            
            
            df_build.reset_index(drop=True, inplace=True)
            df_coords.reset_index(drop=True, inplace=True)
            
            df_build = pd.concat([df_build, df_coords[['x', 'y']]], axis=1)
            
            
            #img_array_1c = make_image_1c(data_2darray = pd.concat([pd.Series(np.ones(df_build.iloc[:,:].shape[0])),df_build.iloc[:,-2:]], axis=1).to_numpy()) 
            img_array_1c = make_image_1c(data_2darray = pd.concat([pd.Series(np.sum(df_build.iloc[:,:-2],axis=1)),df_build.iloc[:,-2:]], axis=1).to_numpy()) 
            self.img_array_1c = img_array_1c

            
            
            ROIs = input("How to name your ROIs, from left to right, top to bottom? Separate ROI names by one space.") #["PC_10", "PC_35", "P6_10", "P6_35", "GF_10", "GF_35", "CMC_10", "CMC_35"]
            ROIs = ROIs.split(" ")

            ROI_exit = False
            while not ROI_exit:
                ROI_dim_array = np.empty((0,4))
                for ROI in ROIs:
                    #ROI_dim = cv2.selectROI('ROI_selection', img_array_1c[0], showCrosshair=True)
                    ROI_select = bbox_select(img_array_1c[0])
                    ROI_select.disconnect_event.wait() 
                    #ROI_dim = np.asarray(ROI_select.selected_points)
                    #ROI_dim_array = np.append(ROI_dim_array, ROI_dim.reshape((1,-1)), axis=0)
               # cv2.destroyAllWindows()
                ROI_sele = input("Keep ROI selecction? (yes/no) ")
                if ROI_sele == "yes":
                    ROI_exit = True
                else:
                    ROI_exit = False
            
            
            df_build['ROI'] = 'placeholder'
            df_build['replicate'] = rep
            #df_build.drop("ROI", axis=1, inplace=True)
            

            for i,ROI in enumerate(ROIs):
                print(f"Select ROI {i}, labeled {ROI}.")
                bottom = ROI_dim_array[i][1]
                top = ROI_dim_array[i][1] + ROI_dim_array[i][3]
                left = ROI_dim_array[i][0]
                right = ROI_dim_array[i][0] + ROI_dim_array[i][2]
            
                ROI_xy = (df_build['x'] >= left) & (df_build['x'] < right) & (df_build['y'] >= bottom) & (df_build['y'] < top)
                df_build["ROI"][ROI_xy] = ROI
                print(f"ROI {i} is selected and labeled {ROI}!")
                
                ROI_edge = pd.concat([ROI_edge, pd.Series([ROI,bottom,top,left,right,rep])], axis=1)

                    
                
            self.df_pixel_all = pd.concat([self.df_pixel_all, df_build])
            
        self.df_pixel_all.reset_index(drop=True, inplace=True)
        self.df_pixel_all.columns = self.df_pixel_all.columns.astype(str)
        ROI_edge = ROI_edge.T.reset_index(drop=True)
        ROI_edge.columns = ["ROI","bottom","top","left","right","replicate"]
        self.ROI_info = ROI_edge
        
        
    def calibrate_mz(self):
        if not os.path.exists(f"{self.data_dir}/figures"):
            os.makedirs(f"{self.data_dir}/figures")
        if not os.path.exists(f"{self.data_dir}/figures/calibration"):
            os.makedirs(f"{self.data_dir}/figures/calibration")
        plt.style.use('classic')
            
            
        exit_regression = False
        while not exit_regression:
            degree = int(input("What is the degree of linear model for calibration? Enter an integer. "))
            reference_mz = input("What are your reference m/z's? ")
            reference_mz = reference_mz.split(" ")
            reference_mz = np.asarray(reference_mz).astype(np.float32)
            resid_index = np.argmin(abs((self.mz.to_numpy()[:, np.newaxis] - reference_mz [np.newaxis,:])), axis=0)
            resid = self.mz[resid_index].to_numpy() - reference_mz
    
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features_resid = poly.fit_transform(self.mz[resid_index].to_numpy().reshape(-1,1))
            poly_reg_model = LinearRegression()
            poly_reg_model.fit(poly_features_resid, resid)
            R2 = poly_reg_model.score(poly_features_resid, resid)
            
            poly_features = poly.fit_transform(self.mz.to_numpy().reshape(-1,1))
            resid_predicted = poly_reg_model.predict(poly_features)
            
            fig, ax = plt.subplots(figsize=(self.fig_ratio, self.fig_ratio*0.7))
            ax.scatter(self.mz[resid_index], self.mz[resid_index] - resid, color="red")
            resid_index_compl = np.ones(self.mz.shape[0], dtype=bool)
            resid_index_compl[resid_index] = False
            ax.scatter(self.mz[resid_index_compl], self.mz[resid_index_compl], color="blue", alpha=0.25)
            ax.plot(self.mz, self.mz + resid_predicted, color='black')
            ax.set_title(f'Linear model of degree {degree}')
            ax.set_xlabel('Experimental m/z')
            ax.set_ylabel('Calibrated m/z')
            ax.text(0.5, 0.95, f'$R^2 = {R2:.3f}$', fontsize=12, horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
            plt.show()
            
            calibration = input("Accept changes to calibration? (yes/no) ")
            if calibration == "yes":
                self.mz -= resid_predicted
                fig.savefig(f"{self.data_dir}/figures/calibration/accepted_calibration_degree{degree}.png")
                exit_regression = True
            elif calibration == "no":
                exit_regression = False
                plt.close(fig)
            else:
                exit_regression = False
                plt.close(fig)
    
        
     
    def MS1_search(self, ppm_threshold=5):
        ion_type = input("What are your ion types of interest? Separate each ion type by one space? ")
        self.ion_type = ion_type.split(" ")
        mass_diff = input("What is the corresponding mass difference of your ion type(s)? List ion types in same order as previously aand separate each ion type by one space. ")
        mass_diff = mass_diff.split(" ")
        self.mass_diff = np.asarray(mass_diff).astype(np.float32)
        
        MS1_db_path = input("What is the file path of your database in csv format? ")
        mz_col = int(input("Which column in your MS1 database corresponds to neurtral monoisotopic masses? ")) - 1
        print("Importing MS1 database. . .")
        MS1_db = pd.read_csv(MS1_db_path)
        print("Finished importing MS1 database!")

        MS1_db.rename(columns={MS1_db.columns[mz_col]: 'exact_mass'}, inplace=True)
        
        for mz_diff, adduct in zip(self.mass_diff, ion_type):#zip(self.mass_diff, self.ion_type):
            MS1 = MS1_db.copy()
            MS1["ion_mass"] = MS1["exact_mass"] + mz_diff
            MS1["ion_type"] = adduct
            
            MS1_ion_mass = MS1["ion_mass"].to_numpy()
            mz = self.mz.copy()
            MS1_hits = pd.DataFrame(abs((mz[:, np.newaxis] - MS1_ion_mass[np.newaxis,:]) / mz[:, np.newaxis] * 10**6) < ppm_threshold)
            MS1_hits = MS1_hits.stack()[MS1_hits.stack() == True].index
            
            all_rows = []
            for i, row in enumerate(MS1_hits):
                combined_row = [row[0]] + MS1.iloc[row[1]].tolist()
                all_rows.append(combined_row)
                
            self.ms1_df = pd.DataFrame(all_rows, columns=[['analyte'] + MS1.columns.to_list()]) 
            
            
    def filter_analytes(self, method="MS1"):
        if method == "MS1":
            truncate_col = np.where(self.df_pixel_all.columns == "x")[0][0]
            self.df_pixel_all = pd.concat([self.df_pixel_all.iloc[:,np.unique(self.ms1_df['analyte'])], self.df_pixel_all.iloc[:,truncate_col:]], axis=1)
            truncate_col = np.where(self.df_pixel_all.columns == "x")[0][0]
            self.df_pixel_all.columns = np.append(np.arange(truncate_col).astype(str), self.df_pixel_all.columns[truncate_col:])
            self.mz = self.mz[np.unique(self.ms1_df['analyte'])]
            self.mz.reset_index(drop=True,inplace=True)
        else:
            print("Method not recognized. Please choose from options specified in documentation. ")
            pass
        
    
    def normalize_pixel(self, normalization = "TIC"):
        print(f"Normalizing pixels with {normalization}.")
        if normalization == "RMS":
            truncate_col = np.where(self.df_pixel_all.columns=='x')[0][0]
            self.df_pixel_all.iloc[:,:truncate_col] = np.sqrt(self.df_pixel_all.iloc[:,:truncate_col]**2 / self.df_pixel_all.shape[1])
        elif normalization == "TIC":
            truncate_col = np.where(self.df_pixel_all.columns=='x')[0][0]
            self.df_pixel_all.iloc[:,:truncate_col] = self.df_pixel_all.iloc[:,:truncate_col].div(np.sum(self.df_pixel_all.iloc[:,:truncate_col],axis=1), axis=0)
        else:
            print("Normalization method not found. Default to no normalization")
            pass
        
    
        
    def get_ion_image(self, replicate=0, show_ROI=True, show_square=False, color_scheme="inferno"):
        if not os.path.exists(f"{self.data_dir}/figures"):
            os.makedirs(f"{self.data_dir}/figures")
        if not os.path.exists(f"{self.data_dir}/figures/ion_image"):
            os.makedirs(f"{self.data_dir}/figures/ion_image")
            
            
        plt.style.use('classic')

        truncate_col = np.where(self.df_pixel_all.columns=='x')[0][0] + 2
        self.df_pixel_all.iloc[:, :truncate_col-2] = self.df_pixel_all.iloc[:, :truncate_col-2].fillna(0)
        insitu_df = pd.concat([self.df_pixel_all['replicate'], 
                               self.df_pixel_all.iloc[:,:truncate_col]], 
                               axis=1)
        insitu_df = insitu_df[insitu_df['replicate']==replicate]
        insitu_df = insitu_df.drop(columns=['replicate'])
        
        if self.jit:
            insitu_2darray = insitu_df.to_numpy()
            img_array_1c = np.zeros([insitu_2darray.shape[1]-2, int(max(insitu_2darray[:,-1]))+1, 
                                                            int(max(insitu_2darray[:,-2]))+1, 1])
            insitu_image = make_image_1c_njit(data_2darray = insitu_2darray, img_array_1c = img_array_1c, remap_coord = False)
            del insitu_2darray, img_array_1c
        else:
            insitu_image = make_image_1c(data_2darray = insitu_df.to_numpy(), remap_coord = False, max_normalize=False)

        ROI_info = self.ROI_info[self.ROI_info["replicate"]==replicate]
        for analyte, im in enumerate(insitu_image[:,:,:,0]):
            plt.style.use('dark_background')
            fig = plt.figure(figsize = np.max(insitu_df[['x','y']], axis=0)/10) #
            plt.rcParams.update({'font.size': np.max(self.df_pixel_all[['x','y']]).max()/10})
            gs = GridSpec(1, 2, width_ratios=[np.max(self.df_pixel_all[['x','y']]).max(), np.max(self.df_pixel_all[['x','y']]).max()/25], wspace=0.05) #image and legend grids
            ax1 = fig.add_subplot(gs[0])
            im_plot = ax1.imshow(im, cmap=color_scheme)
            draw_ROI(ROI_info, show_ROI=show_ROI, show_square=show_square, linewidth=3)
            plt.xticks([])  
            plt.yticks([]) 
            plt.title(f"m/z {self.mz[analyte]:.3f}")
            ax2 = fig.add_subplot(gs[1])
            colorbar = plt.colorbar(im_plot, cax=ax2)
            min_val, max_val = im.min(), im.max()
            colorbar.set_ticks([min_val, max_val])
            colorbar.set_ticklabels([0, 100])
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/figures/ion_image/mz_{self.mz[analyte]:.3f}_replicate{replicate}.png")
            plt.show(fig)
            

        
        
    def make_FC_plot(self, pthreshold=0.05, FCthreshold=1.5):
        if not os.path.exists(f"{self.data_dir}/figures"):
            os.makedirs(f"{self.data_dir}/figures")
        if not os.path.exists(f"{self.data_dir}/figures/volcano"):
            os.makedirs(f"{self.data_dir}/figures/volcano")
        if not os.path.exists(f"{self.data_dir}/figures/volcano_unlabeled"):
            os.makedirs(f"{self.data_dir}/figures/volcano_unlabeled")
            
        plt.style.use('classic')
        
        # prepare dataframe for p-value and FC computations
        self.df_mean_all = self.df_pixel_all.groupby(['ROI', 'replicate']).mean().reset_index()
        self.df_mean_all.columns = self.df_mean_all.columns.astype(str)
        self.df_mean_all = self.df_mean_all[self.df_mean_all['ROI'] != 'placeholder']
        df_mean_all_long = pd.melt(self.df_mean_all, id_vars=['ROI', 'replicate', 'x', 'y'], var_name='analyte', value_name='intensity')
    
            
        
        df_FC = df_mean_all_long.groupby(['ROI', 'analyte']).mean().reset_index()
        
        print("Starting to generate volcano plots.")
        for combo in list(permutations(df_FC['ROI'].unique(), 2)):
            df_FC_pair = df_FC[(df_FC['ROI'] == combo[0]) | (df_FC['ROI'] == combo[1])]
            
            reference_intensity = df_FC_pair[df_FC_pair['ROI'] == combo[1]].groupby('analyte')['intensity'].mean()
            df_FC_pair = df_FC_pair.merge(reference_intensity, on='analyte', suffixes=('', '_ref'))
            df_FC_pair = df_FC_pair[(df_FC_pair[['intensity_ref']] != 0).all(axis=1)]
            df_FC_pair['intensity'] /= df_FC_pair['intensity_ref'] 
            
            df_FC_pair['intensity'] = np.log2(df_FC_pair['intensity'])

            df_FC_pair['p'] = 1
            # compute p-value for each analyte
            for analyte_sele in df_FC_pair['analyte'].unique():
                df_analyte = df_mean_all_long[df_mean_all_long['analyte']==analyte_sele]
                lm_df = ols('intensity ~ C(ROI)', 
                            data=df_analyte).fit() 
                anova_df = sm.stats.anova_lm(lm_df, typ=1) 
                
                df_FC_pair.loc[df_FC_pair['analyte']==analyte_sele, 'p'] = anova_df['PR(>F)'][0]
                
            df_FC_pair['p'] = -np.log10(df_FC_pair['p'])
            
          
            df_FC_pair['condition'] = "none"
            df_FC_pair['condition'].loc[(df_FC_pair['p'] > -np.log10(pthreshold)) & (df_FC_pair['intensity'] < -np.log2(FCthreshold))] = "down"
            df_FC_pair['condition'].loc[(df_FC_pair['p'] > -np.log10(pthreshold)) & (df_FC_pair['intensity'] > np.log2(FCthreshold))] = "up"
            
            palette = {
                        "none": "grey",   # Grey for non-significant points
                        "down": "blue",   # Blue for downregulated
                        "up": "red"       # Red for upregulated
                      }
            
        
            plt.figure(figsize = (self.fig_ratio*1.5,self.fig_ratio*1.5)) # larger figure may be better
            plt.rcParams.update({'font.size': 200/self.fig_ratio*1.5})
            sns.scatterplot(data=df_FC_pair[df_FC_pair['ROI'] == combo[0]],
                            x="intensity", y="p", hue="condition", 
                            palette=palette, s=80)
            plt.title(f'{combo[0]} / {combo[1]}')
            plt.xlabel('Log2 FC')
            plt.ylabel('-Log10 p_value')
            plt.axvline(x = -np.log2(FCthreshold), color = 'black', linewidth=1.5, linestyle='--')
            plt.axvline(x = np.log2(FCthreshold), color = 'black', linewidth=1.5, linestyle='--')
            plt.axhline(y = -np.log10(pthreshold), color = 'black', linewidth=1.5, linestyle='--')
            
            for _, row in df_FC_pair[df_FC_pair['ROI'] == combo[0]].iterrows():
                if row['p'] > -np.log10(pthreshold) and (row['intensity'] > np.log2(FCthreshold) or row['intensity'] < -np.log2(FCthreshold)):
                    plot_mz = f"{self.mz.iloc[int(row['analyte'])]:.1f}"
                    plt.text(row['intensity'], row['p'], plot_mz, 
                             horizontalalignment='left', size=15, color='black', weight='semibold',
                             alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/figures/volcano/{combo[0]}_{combo[1]}", dpi=300)  
            plt.show()
            
            #vocalno unlabeled
            plt.figure(figsize = (self.fig_ratio*1.5,self.fig_ratio*1.5)) # larger figure may be better
            plt.rcParams.update({'font.size': 200/self.fig_ratio*1.5})
            sns.scatterplot(data=df_FC_pair[df_FC_pair['ROI'] == combo[0]],
                            x="intensity", y="p", hue="condition", 
                            palette=palette, s=80)
            plt.title(f'{combo[0]} / {combo[1]}')
            plt.xlabel('Log2 FC')
            plt.ylabel('-Log10 p_value')
            plt.axvline(x = -np.log2(FCthreshold), color = 'black', linewidth=1.5, linestyle='--')
            plt.axvline(x = np.log2(FCthreshold), color = 'black', linewidth=1.5, linestyle='--')
            plt.axhline(y = -np.log10(pthreshold), color = 'black', linewidth=1.5, linestyle='--')
            
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/figures/volcano_unlabeled/{combo[0]}_{combo[1]}", dpi=300)  
            plt.show()
            
            print(f"Finished generating volcano plots. \n\n Volcano plots were stroed in folder {self.data_dir}/figures.")
     
        #===heatmap===#
        plt.figure(figsize = (len(list(permutations(df_FC['ROI'].unique(), 2)))*3,len(df_FC['analyte'].unique())/1.5))
        plt.rcParams.update({'font.size': len(df_FC['analyte'].unique())/2.5})
        df_hm = df_FC.pivot(index="analyte", columns="ROI", values="intensity")
        df_hm = np.log2(df_hm.div(df_hm.mean(axis=1), axis=0))
        sns.heatmap(df_hm, annot=True, fmt=".1f", vmin=-3, vmax=3, cbar_kws={"shrink": 0.5, "label":"log2 FC"})
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/figures/volcano/heatmap", dpi=100)  
        plt.close()
        

    def insitu_clustering(self, k=10, perplexity=15, replicate=0, show_ROI=True, show_square=True):
        if not os.path.exists(f"{self.data_dir}/figures"):
            os.makedirs(f"{self.data_dir}/figures")
        if not os.path.exists(f"{self.data_dir}/figures/insitu_cluster"):
            os.makedirs(f"{self.data_dir}/figures/insitu_cluster")

            
        plt.style.use('classic')
        
        #KMeans
        self.k = k
        self.perplexity = perplexity
        
        truncate_col = np.where(self.df_pixel_all.columns=='x')[0][0]
        self.df_pixel_all.iloc[:, :truncate_col] = self.df_pixel_all.iloc[:, :truncate_col].fillna(0)
        df_pixel_all_max_all = self.df_pixel_all.copy()
        df_pixel_all_max_all.iloc[:,:truncate_col] = df_pixel_all_max_all.iloc[:,:truncate_col].div(np.max(df_pixel_all_max_all.iloc[:,:truncate_col],axis=0))
        
        if self.gpu == False:
            print("Spatial segmentation on CPU. Computing time may be long.")
            from sklearn.cluster import KMeans
            from sklearn.manifold import TSNE
            k_means = KMeans(n_clusters=self.k, init = "k-means++", random_state=0, n_init="auto").fit(df_pixel_all_max_all.iloc[:,:truncate_col])
            kmeans_labels = k_means.labels_ + 1
    
            #tSNE
            num_iters = 1000
            my_random_state = 1
            tsne = TSNE(n_components = 2, n_iter = num_iters,
                          perplexity = self.perplexity, random_state = my_random_state) 
            tsne_embedding = tsne.fit_transform(df_pixel_all_max_all.iloc[:,:truncate_col])
            
        elif self.gpu == True:
            try:
                print("Spatial segmentation on GPU.")
                from cuml.manifold import TSNE
                from cuml.cluster import KMeans
                from cuml import using_output_type
    
                with using_output_type('numpy'):
                    #kmeans
                    k_means = KMeans(n_clusters = self.k).fit(df_pixel_all_max_all.iloc[:,:truncate_col])
                    kmeans_labels = k_means.labels_ + 1
                    
                    # tsne
                    num_iters = 1000
                    my_random_state = 1
                    tsne = TSNE(n_components = 2, n_iter = num_iters,
                                  perplexity = self.perplexity, random_state = my_random_state,
                                  n_neighbors = 150)
                    tsne_embedding = tsne.fit_transform(df_pixel_all_max_all.iloc[:,:truncate_col])
            except:
                    print("GPU error. Defaulting to CPU." )
                    print("Spatial segmentation on CPU. Computing time may be long.")
                    from sklearn.cluster import KMeans
                    from sklearn.manifold import TSNE
                    k_means = KMeans(n_clusters=self.k, init = "k-means++", random_state=0, n_init="auto").fit(df_pixel_all_max_all.iloc[:,:truncate_col])
                    kmeans_labels = k_means.labels_ + 1
            
                    #tSNE
                    num_iters = 1000
                    my_random_state = 1
                    tsne = TSNE(n_components = 2, n_iter = num_iters,
                                  perplexity = self.perplexity, random_state = my_random_state) 
                    tsne_embedding = tsne.fit_transform(df_pixel_all_max_all.iloc[:,:truncate_col])

        df_pixel_all_max_all['insitu_cluster'] = pd.DataFrame(kmeans_labels)

        plt.figure(figsize = (self.fig_ratio*1.5,self.fig_ratio*1.5)) # larger figure may be better
        plt.rcParams.update({'font.size': 200/self.fig_ratio*1.5})
        cmap = plt.cm.get_cmap('rainbow', len(np.unique(kmeans_labels)))
        for i, label in enumerate(np.unique(kmeans_labels)):
            plt.scatter(tsne_embedding[kmeans_labels == label, 0], tsne_embedding[kmeans_labels == label, 1], 
                        c=cmap(i),  # Normalize color within the colormap range
                        alpha=0.6, label=f'Cluster {label}')

        plt.title("k-Means Clusters of Global Molecular Profile in t-SNE Embedding")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        legend = plt.legend(title='Cluster')
        for legend_handle in legend.legendHandles: #reset legend alpha
            legend_handle.set_alpha(1)
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/figures/insitu_cluster/kmeans_tsne", dpi=100)  


        categories = pd.Categorical(df_pixel_all_max_all['ROI'])
        unique_categories = categories.categories
        cmap = plt.cm.get_cmap('rainbow', len(unique_categories))  # Get a colormap with as many colors as there are categories

        plt.figure(figsize = (self.fig_ratio*1.5,self.fig_ratio*1.5))
        plt.rcParams.update({'font.size': 200/self.fig_ratio*1.5})
        # Plot each category separately to set up legends correctly
        for i, category in enumerate(unique_categories[unique_categories!="placeholder"]):
            mask = df_pixel_all_max_all['ROI'] == category
            plt.scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1], 
                        color=cmap(i), alpha=0.6, label=category)

        plt.title("Global Molecular Profile in t-SNE Embedding")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

        # Adding the legend
        legend = plt.legend(title='ROI')
        for legend_handle in legend.legendHandles: #reset legend alpha
            legend_handle.set_alpha(1)
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/figures/insitu_cluster/original_space_tsne", dpi=100)  

        #===========IN SITU============# 
        insitu_df = pd.concat([df_pixel_all_max_all['replicate'],pd.DataFrame(kmeans_labels,columns=["kmeans_label"]), 
                               df_pixel_all_max_all[['x','y']]], axis=1) 
        insitu_df = insitu_df[insitu_df['replicate']==replicate]
        insitu_df = insitu_df.drop(columns=['replicate'])

        insitu_image = make_image_1c(data_2darray = insitu_df.to_numpy(), remap_coord = False, max_normalize=False)
        insitu_image = np.ma.masked_equal(insitu_image, 0)

        ROI_info = self.ROI_info[self.ROI_info["replicate"]==replicate]
        plt.style.use('dark_background')
        fig = plt.figure(figsize = np.max(insitu_df[['x','y']], axis=0)/10) #
        plt.rcParams.update({'font.size': np.max(df_pixel_all_max_all[['x','y']]).max()/10})
        gs = GridSpec(1, 2, width_ratios=[np.max(df_pixel_all_max_all[['x','y']]).max(), np.max(df_pixel_all_max_all[['x','y']]).max()/25], wspace=0.05) #image and legend grids
        ax1 = fig.add_subplot(gs[0])
        kmeans_insitu = ax1.imshow(insitu_image[0], cmap="rainbow")
        draw_ROI(ROI_info, show_ROI=show_ROI, show_square=show_square, linewidth=3)
        plt.xticks([])  
        plt.yticks([]) 
        plt.title("")
        ax2 = fig.add_subplot(gs[1])
        colorbar = plt.colorbar(kmeans_insitu, cax=ax2)
        min_val, max_val = np.min(insitu_image[0]), np.max(insitu_image[0])
        colorbar.set_ticks(range(int(min_val), int(max_val) + 1))  
        plt.tight_layout()

        plt.savefig(f"{self.data_dir}/figures/insitu_cluster/kmeans_insitu_image", dpi=200) 
        plt.show(fig)
        

        
    def image_clustering(self, k=10, perplexity=5, replicate=0, show_ROI=True, show_square=True, color_scheme="inferno", zoom=0.1, quantile=99):
        if not os.path.exists(f"{self.data_dir}/figures"):
            os.makedirs(f"{self.data_dir}/figures")
        if not os.path.exists(f"{self.data_dir}/figures/image_cluster"):
            os.makedirs(f"{self.data_dir}/figures/image_cluster")
            
            
        plt.style.use('classic')
        
        #KMeans
        self.k = k
        self.perplexity = perplexity
        
        truncate_col = np.where(self.df_pixel_all.columns=='x')[0][0]
        self.df_pixel_all.iloc[:, :truncate_col] = self.df_pixel_all.iloc[:, :truncate_col].fillna(0)
        df_pixel_all_max = self.df_pixel_all.copy()
        df_pixel_all_max.iloc[:,:truncate_col] = df_pixel_all_max.iloc[:,:truncate_col].div(np.max(df_pixel_all_max.iloc[:,:truncate_col],axis=0))
        
        if self.gpu == False:
            print("Image clustering on CPU. Computing time may be long.")
            from sklearn.cluster import KMeans
            from sklearn.manifold import TSNE
            
            k_means = KMeans(n_clusters=self.k, init = "k-means++", random_state=0, n_init="auto").fit(df_pixel_all_max.iloc[:,:truncate_col].T)
            kmeans_labels = k_means.labels_ + 1
    
            #tSNE
            num_iters = 1000
            my_random_state = 1
            tsne = TSNE(n_components = 2, n_iter = num_iters,
                          perplexity = self.perplexity, random_state = my_random_state) 
            tsne_embedding = tsne.fit_transform(df_pixel_all_max.iloc[:,:truncate_col].T)
            
        elif self.gpu == True:
            
            try:
                print("Image clustering on GPU.")
                from cuml.manifold import TSNE
                from cuml.cluster import KMeans
                from cuml import using_output_type
    
                with using_output_type('numpy'):
                    
                    
                    k_means = KMeans(n_clusters = self.k).fit(df_pixel_all_max.iloc[:,:truncate_col].T)
                    kmeans_labels = k_means.labels_ + 1
                    
                    # tsne
                    num_iters = 1000
                    my_random_state = 1
                    tsne = TSNE(n_components = 2, n_iter = num_iters,
                                  perplexity = self.perplexity, random_state = my_random_state,
                                  n_neighbors = 150)
                    tsne_embedding = tsne.fit_transform(df_pixel_all_max.iloc[:,:truncate_col].T)
            except:
                print("GPU error. Defaulting to CPU." )
                print("Image clustering on CPU. Computing time may be long.")
                from sklearn.cluster import KMeans
                from sklearn.manifold import TSNE
                
                k_means = KMeans(n_clusters=self.k, init = "k-means++", random_state=0, n_init="auto").fit(df_pixel_all_max.iloc[:,:truncate_col].T)
                kmeans_labels = k_means.labels_ + 1
        
                #tSNE
                num_iters = 1000
                my_random_state = 1
                tsne = TSNE(n_components = 2, n_iter = num_iters,
                              perplexity = self.perplexity, random_state = my_random_state) 
                tsne_embedding = tsne.fit_transform(df_pixel_all_max.iloc[:,:truncate_col].T)
            
        image_cluster_label = pd.DataFrame(kmeans_labels)
        image_cluster_label = image_cluster_label.reset_index()
        image_cluster_label.rename(columns={'index': 'analyte', 0: 'cluster'}, inplace=True)
        
        image_df = df_pixel_all_max.groupby(['ROI', 'replicate']).mean().reset_index()
        cluster_mean_df = image_df['ROI']
        image_cluster_all = self.df_pixel_all['ROI']
        for cluster_index in range(1, k+1): #loop through clusters
            cluster_df = image_df[image_cluster_label[image_cluster_label['cluster']==cluster_index]['analyte'].to_numpy().astype(str)].mean(axis=1)
            cluster_df.name = str(cluster_index)
            cluster_mean_df = pd.concat([cluster_mean_df, cluster_df], axis=1)
            
            image_cluster = df_pixel_all_max[image_cluster_label[image_cluster_label['cluster']==cluster_index]['analyte'].to_numpy().astype(str)].mean(axis=1)
            image_cluster.name = str(cluster_index)
            image_cluster_all = pd.concat([image_cluster_all, image_cluster], axis=1)
            
        cluster_mean_df = cluster_mean_df[cluster_mean_df['ROI']!="placeholder"]
        
        image_cluster_all[['x','y']] = self.df_pixel_all[['x','y']]
        image_cluster_all = image_cluster_all[self.df_pixel_all['replicate']==replicate]
        image_cluster_all = image_cluster_all[image_cluster_all['ROI']!="placeholder"]
                
        # kmeans image clusters in t-SNE embedding
        plt.figure(figsize = (self.fig_ratio*1.5,self.fig_ratio*1.5)) # larger figure may be better
        plt.rcParams.update({'font.size': 200/self.fig_ratio*1.5})
        cmap = plt.cm.get_cmap('rainbow', len(np.unique(kmeans_labels)))
        for i, label in enumerate(np.unique(kmeans_labels)):
            plt.scatter(tsne_embedding[kmeans_labels == label, 0], tsne_embedding[kmeans_labels == label, 1], 
                        c=cmap(i),  # Normalize color within the colormap range
                        alpha=1, label=f'Cluster {label}', s=50)
        plt.title("k-Means Clusters of Global Molecular Profile in t-SNE Embedding")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        legend = plt.legend(title='Cluster')
        for legend_handle in legend.legendHandles: #reset legend alpha
            legend_handle.set_alpha(1)
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/figures/image_cluster/image_kmeans_tsne", dpi=100)  
        
        # insitu images in t-SNE embedding
        truncate_col = np.where(self.df_pixel_all.columns=='x')[0][0] + 2
        insitu_df = pd.concat([self.df_pixel_all['replicate'], 
                               df_pixel_all_max.iloc[:,:truncate_col]], 
                               axis=1)
        insitu_df = insitu_df[insitu_df['replicate']==0]
        insitu_df = insitu_df.drop(columns=['replicate'])
        
        if self.jit:
            insitu_2darray = insitu_df.to_numpy()
            img_array_1c = np.zeros([insitu_2darray.shape[1]-2, int(max(insitu_2darray[:,-1]))+1, 
                                                            int(max(insitu_2darray[:,-2]))+1, 1])
            #insitu_2darray[:,:-2] = max_normalize(insitu_2darray[:,:-2])
            insitu_image = make_image_1c_njit(data_2darray = insitu_2darray, img_array_1c = img_array_1c, remap_coord = False)
            del insitu_2darray, img_array_1c
        else:
            insitu_image = make_image_1c(data_2darray = insitu_df.to_numpy(), remap_coord = False, max_normalize=False)
        
        plt.figure(figsize = (self.fig_ratio*1.5,self.fig_ratio*1.5)) # larger figure may be better
        plt.rcParams.update({'font.size': 200/self.fig_ratio*1.5})
        plt.scatter(*tsne_embedding.T)
        for im, xy in zip(insitu_image[:,:,:,0], tsne_embedding):
            q_max = np.percentile(im, quantile)
            norm = Normalize(vmin=im.min(), vmax=q_max)
            im = norm(im)
            im = np.clip(im, 0, 1)
            plot_image_at_point(im, xy, zoom = zoom, color_scheme="inferno")
            plt.title("Images in t-SNE Embedding")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/figures/image_cluster/image_insitu_tsne")
        
        #cluster boxplot
        image_cluster_df_long = pd.melt(cluster_mean_df, id_vars=['ROI'], var_name='cluster', value_name='intensity')
        image_cluster_df_long['intensity'] = image_cluster_df_long['intensity'].div(image_cluster_df_long.groupby(['cluster']).transform('max')['intensity'])
    

        ROI = image_cluster_df_long["ROI"]
        for cluster in image_cluster_df_long['cluster'].unique():
            image_cluster_df_cluster = image_cluster_df_long[image_cluster_df_long['cluster']==cluster]
            pvalue = [
                        sm.stats.anova_lm(
                            ols('intensity ~ C(ROI)', data=image_cluster_df_cluster.loc[
                                (image_cluster_df_cluster['ROI'] == combo[0]) | 
                                (image_cluster_df_cluster['ROI'] == combo[1]), 
                                ['ROI', 'intensity']]
                            ).fit(),
                            typ=1
                        )['PR(>F)'][0]
                    
                        for combo in list(combinations(ROI.unique(), 2))
                    ]
            model_pairs =   [
                                combo
                                for combo in list(combinations(ROI.unique(), 2))
                            ]
                    
    
            fig = plt.figure(figsize=((10+math.comb(self.ROI_num,2)/10)/2, (7+math.comb(self.ROI_num,2)/3)/2))
            plt.rcParams.update({'font.size': (10+math.comb(self.ROI_num,2)/10)})
            ax = fig.add_subplot()
            sns.boxplot(x="ROI", y="intensity", data=image_cluster_df_cluster, color='white')
            sns.stripplot(x="ROI", y="intensity", data=image_cluster_df_cluster, color='red', size=4, jitter=True)
            
            annotator = Annotator(ax, model_pairs, x="ROI", y="intensity", data=image_cluster_df_cluster)
            formatted_pvalues = [f'{significance(p)}' for p in pvalue]
            annotator.set_custom_annotations(formatted_pvalues)
            annotator.annotate()
            
            plt.title(f'Mean image of cluster {cluster}')
            plt.xlabel('ROI')
            plt.ylabel('Mean Intensity')
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/figures/image_cluster/boxplot_cluster{cluster}.jpg", dpi=200)   
        

        image_cluster_all = image_cluster_all.drop(columns=["ROI"])
        if self.jit:
            insitu_2darray = image_cluster_all.to_numpy()
            img_array_1c = np.zeros([insitu_2darray.shape[1]-2, int(max(insitu_2darray[:,-1]))+1, 
                                                            int(max(insitu_2darray[:,-2]))+1, 1])
            insitu_image = make_image_1c_njit(data_2darray = insitu_2darray, img_array_1c = img_array_1c, remap_coord = False)
            del insitu_2darray, img_array_1c
        else:
            insitu_image = make_image_1c(data_2darray = image_cluster_all.to_numpy(), 
                                         remap_coord = False, max_normalize=False)
            
        ROI_info = self.ROI_info[self.ROI_info["replicate"]==replicate]
        for analyte, im in enumerate(insitu_image[:,:,:,0]):
            plt.style.use('dark_background')
            fig = plt.figure(figsize = np.max(insitu_df[['x','y']], axis=0)/10) #
            plt.rcParams.update({'font.size': np.max(self.df_pixel_all[['x','y']]).max()/10})
            gs = GridSpec(1, 2, width_ratios=[np.max(self.df_pixel_all[['x','y']]).max(), np.max(self.df_pixel_all[['x','y']]).max()/25], wspace=0.05) #image and legend grids
            ax1 = fig.add_subplot(gs[0])
            im_plot = ax1.imshow(im, cmap=color_scheme)
            draw_ROI(ROI_info, show_ROI=show_ROI, show_square=show_square, linewidth=3)
            plt.xticks([])  
            plt.yticks([]) 
            plt.title(f"Mean image of cluster {analyte+1}")
            ax2 = fig.add_subplot(gs[1])
            colorbar = plt.colorbar(im_plot, cax=ax2)
            min_val, max_val = im.min(), im.max()
            colorbar.set_ticks([min_val, max_val])
            colorbar.set_ticklabels([0, 100])
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/figures/image_cluster/mean_image_cluster{analyte+1}.png")
            plt.show(fig)
        del image_cluster_all
        
        

        
    def make_boxplot(self):
        if not os.path.exists(f"{self.data_dir}/figures"):
            os.makedirs(f"{self.data_dir}/figures")
        if not os.path.exists(f"{self.data_dir}/figures/boxplot"):
            os.makedirs(f"{self.data_dir}/figures/boxplot")
            


        self.df_mean_all = self.df_pixel_all.groupby(['ROI', 'replicate']).mean().reset_index()
        self.df_mean_all.columns = self.df_mean_all.columns.astype(str)
        self.df_mean_all = self.df_mean_all[self.df_mean_all['ROI'] != 'placeholder']
        ROI = self.df_mean_all.iloc[:,0]
        df_mean_all_long = pd.melt(self.df_mean_all, id_vars=['ROI', 'replicate', 'x', 'y'], var_name='analyte', value_name='intensity')
        for analyte in df_mean_all_long['analyte'].unique():
            df_mean_all_analyte = df_mean_all_long[df_mean_all_long['analyte']==analyte]
            pvalue = [
                        sm.stats.anova_lm(
                            ols('intensity ~ C(ROI)', data=df_mean_all_analyte.loc[
                                (df_mean_all_analyte['ROI'] == combo[0]) | 
                                (df_mean_all_analyte['ROI'] == combo[1]), 
                                ['ROI', 'intensity']]
                            ).fit(),
                            typ=1
                        )['PR(>F)'][0]
                    
                        for combo in list(combinations(ROI.unique(), 2))
                    ]
            model_pairs =   [
                                combo
                                for combo in list(combinations(ROI.unique(), 2))
                            ]
                    
    
            fig = plt.figure(figsize=(10+math.comb(self.ROI_num,2)/10, 7+math.comb(self.ROI_num,2)/3))
            ax = fig.add_subplot()
            plt.style.use('classic')   
            plt.rcParams.update({'font.size': (7+math.comb(self.ROI_num,2))})

            sns.boxplot(x="ROI", y="intensity", data=df_mean_all_analyte, color='white')
            sns.stripplot(x="ROI", y="intensity", data=df_mean_all_analyte, color='red', size=4, jitter=True)
            
            annotator = Annotator(ax, model_pairs, x="ROI", y="intensity", data=df_mean_all_analyte)
            formatted_pvalues = [f'{significance(p)}' for p in pvalue]
            annotator.set_custom_annotations(formatted_pvalues)
            annotator.annotate()
            
            plt.title(f'm/z {self.mz.iloc[int(analyte)]}')
            plt.xlabel('ROI')
            plt.ylabel('Mean Intensity')
            plt.tight_layout()
            plt.savefig(f"{self.data_dir}/figures/boxplot/mz_{self.mz[int(analyte)]:.3f}.jpg", dpi=200)   
        



    



# visualize_data = DataAnalysis()
# visualize_data.load_preprocessed_data() # Z:\Lin\2024\python\Troy_Silverman\05-03-2024
# visualize_data.normalize_pixel(normalization = "RMS")
# visualize_data.get_ion_image()
# visualize_data.make_FC_plot(pthreshold=0.05, FCthreshold=1.5)
# visualize_data.insitu_clustering(k=5, perplexity=20)
# visualize_data.image_clustering(k=5, perplexity=20)
# visualize_data.make_boxplot()





