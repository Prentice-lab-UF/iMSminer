# -*- coding: utf-8 -*-
"""
@author: Yu Tin Lin (yutinlin@stanford.edu)
@author: Haohui Bao (susanab20020911@gmail.com)
@author: Boone M. Prentice (booneprentice@ufl.chem.edu)
"""

# ==== TO DO LIST ======#
# \item modularize conditional statements and apply decoration functions
# \item provide function descriptions
# \item document codes
print("s1")
import os
os.chdir("/content/drive/My Drive/Colab Notebooks/iMSminer_colab")
from ImzMLParser_chunk import ImzMLParser_chunk
from assorted_functions import get_chunk_ms_info, get_chunk_ms_info2, get_chunk_ms_info_inhomogeneous, chunk_prep, chunk_prep_inhomogeneous, integrate_peak, get_p2, get_spectrum, Aligner_CPU
import gc
import time
print("s2")
import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.signal import find_peaks, peak_widths
print("s3")
from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt
import ray
from bokeh.plotting import figure, show, save, output_file
print("s4")

if not ray.is_initialized():
    ray.init(num_cpus=10)


try:
    os.chdir("/content/drive/My Drive/Colab Notebooks/iMSminer_colab")
    import cupy as cp
    from assorted_functions import Aligner_GPU
except:
    pass


# ========================DATA PROCESSING========================#

class Preprocess:
    """Contains functions to import imzML, generate interactive mean mass spectrum, perform peak picking, mass alignment, and peak integration
    """

    def __init__(
            self
    ):

        # Z:\Lin\2023\Nor_Test2_imzML\2022-12-15_04.imzML   Z:\Scoggins\MC\2022-09-16 Image\2022-09-16 MC Image.imzML
        self.directory = input(
            "What is the directory of imzML files to process? ")
        # Z:\Lin\2023\Python\test
        self.data_dir = input("Where to save the spectral data? ")

        test_library = ['cupy']
        for lib in test_library:
            try:
                __import__(lib)
                print(f"{lib.capitalize()} is installed and imported successfully!")
                self.gpu = True
            except ImportError:
                print(
                    f"{lib.capitalize()} is not installed or could not be imported.")
                self.gpu = False
                break

    def peak_pick(self, percent_RAM=5, method="point", rel_height=0.9, generate_spectrum=True):
        """Perform peak picking to locate signals above a defined LOQ (k * noise) by specifying k, calculating noise, and at specified minimum distance between peaks 

        Parameters
        ----------
        percent_RAM : int, optional
            percent available RAM occupied by chunk, by default 5
        method : str, optional
            method of computing noise, by default "point"
            method `point` takes specified lower and upper m/z bound of a region in spectrum without signals and compute its standard deviation to define noise level
            method `specify_noise` takes user-specified noise level
            method `automatic` computes standard deviation of non-outliers (< k * z_score), where k is specified by user
            method `binning` re-bins mass spectra (userful for compressed data with inhomogeneous shapes), then computes noise using method "automatic"
        rel_height : float, optional
            peak height cutoff for peak integration, by default 0.9
        generate_spectrum : bool, optional
            render average spectrum to directory specified by user, by default True
        lwr : float
            lower m/z bound of a region in spectrum without signals
        upr : float
            upper m/z bound of a region in spectrum without signals
        dist : int
            minimum number of datapoints for peak separation
        loq : float
            number of times the noise level (k * noise) to define limit of quantification used in peak picking
        rp_factor : float
            in method `binning`, factor to scale number of bins; affects mass resolution  
        resolution_progress : str
            sets resolution for binning if user specifies `yes`
        """

        datasets = os.listdir(self.directory)
        datasets = np.asarray(datasets)[(np.char.find(datasets, "ibd") == -1)]

        p = ImzMLParser_chunk(f"{self.directory}/{datasets[0]}")
        dtype = np.float32
        num_chunks, chunk_size_base, chunk_start, remainder = chunk_prep(
            p, percent_RAM)

        try:
            if method == "point":
                self.lwr = float(
                    input("What is the lower m/z bound of noise? "))
                self.upr = float(
                    input("What is the upper m/z bound of noise? "))
                self.dist = int(
                    input("What is the minimum number of data points between peaks? "))
                self.loq = float(input(
                    "What is your limit of quantification? Enter a coefficient k such that LOQ = k * noise. "))

                p2, p2_width, noise, mzs, avg_spectrum = get_p2(p=p, loq=self.loq, lwr=self.lwr, upr=self.upr, rel_height=rel_height,
                                                                chunk_size_base=chunk_size_base, num_chunks=num_chunks,
                                                                chunk_start=chunk_start, remainder=remainder, dist=self.dist,
                                                                method=method)
            elif method == "specify_noise":
                self.lwr = float("NaN")
                self.upr = float("NaN")
                self.noise = float(input(
                    "What is your noise level? Enter a number noise such that LOQ = k * noise. "))
                self.loq = float(input(
                    "What is your limit of quantification? Enter a coefficient k such that LOQ = k * noise. "))
                self.dist = int(
                    input("What is the minimum number of data points between peaks? "))
                p2, p2_width, noise, mzs, avg_spectrum = get_p2(p=p, loq=self.loq, lwr=self.lwr, upr=self.upr, rel_height=rel_height,
                                                                chunk_size_base=chunk_size_base, num_chunks=num_chunks,
                                                                chunk_start=chunk_start, remainder=remainder, dist=self.dist,
                                                                method=method)

            else:
                return print("Method not recognized. Please choose from options specified in documentation. ")

            self.inhomogeneous = False
            self.p2 = p2
            self.p2_width = p2_width
            self.noise = noise
            self.mzs = mzs
            self.avg_spectrum = avg_spectrum

            get_spectrum(output_filepath=f'{self.data_dir}/{datasets[0][:-6]}_avg_spectrum.html',
                         mzs=self.mzs, avg_intensity=avg_spectrum, p2=self.p2)

        except:
            print("Data has inhomogeneous shape. Default to binning.")
            self.RP = float(input("What is your max FWHF? "))
            self.mz_RP = float(input("At what m/z is FWHF calculated? "))
            self.loq = float(input(
                "What is your limit of quantification? Enter a coefficient k such that LOQ = k * noise. "))
            self.dist = int(
                input("What is the minimum number of data points between peaks? "))

            p = ImzMLParser(f"{self.directory}/{datasets[0]}")

            min_mz = 10**10
            max_mz = 0
            for i in range(len(p.coordinates)):
                mz = p.getspectrum(i)[0]
                if min(mz) < min_mz:
                    min_mz = min(mz)
                if max(mz) > max_mz:
                    max_mz = max(mz)

            self.min_mz = min_mz
            self.max_mz = max_mz

            self.rp_factor = 1  # scales number of bins
            resolution_exit = False
            while not resolution_exit:

                avg_spectrum = np.zeros(
                    [int((max_mz-min_mz)/(self.mz_RP/self.RP) * self.rp_factor)+10])

                for i in range(len(p.coordinates)):
                    spectrum_i = p.getspectrum(i)
                    intensity_index = np.round(
                        (spectrum_i[0] - min_mz) / (self.mz_RP/self.RP) * self.rp_factor, 0).astype(np.int64)
                    avg_spectrum[intensity_index] += spectrum_i[1]

                avg_spectrum /= len(p.coordinates)

                self.noise = np.mean(
                    avg_spectrum[abs(scipy.stats.zscore(avg_spectrum)) < 3])

                p2 = find_peaks(x=avg_spectrum, height=self.loq *
                                self.noise, distance=self.dist)
                p2_width = peak_widths(
                    avg_spectrum, p2[0], rel_height=rel_height)

                self.p2 = p2[0]
                self.p2_width = np.asarray(p2_width)
                self.mzs = np.arange(min_mz, max_mz, (max_mz-min_mz) /
                                     (int((max_mz-min_mz)/(self.mz_RP/self.RP)*self.rp_factor)+10))
                self.avg_spectrum = avg_spectrum
                self.inhomogeneous = True

                get_spectrum(output_filepath=f'{self.data_dir}/{datasets[0][:-6]}_avg_spectrum.html',
                             mzs=self.mzs, avg_intensity=avg_spectrum, p2=self.p2)

                resolution_progress = input(
                    "Satisfied with resolution? (yes/no) ")
                if resolution_progress == "yes":
                    resolution_exit = True
                else:
                    resolution_exit = False
                    self.rp_factor = float(
                        input("Scale number of bins by a number: "))
                    self.dist = int(
                        input("What is the minimum number of data points between peaks? "))

    def run(self, percent_RAM=5, peak_alignment=False, align_halfwidth=10, grid_iter_num=100, align_reduce=True, reduce_halfwidth=10, plot_aligned_peak=True, index_peak_plot=0, plot_num_peaks=10):
        """imports imzML files and perform peak-picking, mass alignment, and peak integration

        Parameters
        ----------
        percent_RAM : int, optional
            percent available RAM occupied by chunk, by default 5
        peak_alignment : bool, optional
            performs peak alignment if True, by default False
        align_halfwidth : int, optional
            _description_, by default 10
        grid_iter_num : int, optional
            _description_, by default 100
        align_reduce : bool, optional
            reduce intensity matrix involved in peak alignment if True, by default True
        reduce_halfwidth : int, optional
            range of intensity bins to be kept specified in half-widths if align_reduce, by default 10
        plot_aligned_peak : bool, optional
            render a figure to show peak alignment results if True, by default True
        index_peak_plot : int, optional
            peak with specified analyte index to visualize if plot_aligned_peak, by default 0
        plot_num_peaks : int, optional
            number of peaks (spectra) to plot if plot_aligned_peak, by default 10
        """
        datasets = os.listdir(self.directory)
        datasets = np.asarray(datasets)[(np.char.find(datasets, "ibd") == -1)]
        for dataset in datasets:
            if not self.inhomogeneous:
                p = ImzMLParser_chunk(f"{self.directory}/{dataset}")
                num_chunks, chunk_size_base, chunk_start, remainder = chunk_prep(
                    p, percent_RAM)
            else:
                p = ImzMLParser(f"{self.directory}/{dataset}")
                num_chunks, chunk_size_base, chunk_start, remainder = chunk_prep_inhomogeneous(
                    p, percent_RAM, self.max_mz, self.min_mz, self.mz_RP, self.RP, self.dist)

            p2_new = np.array([])
            info = np.array([])
            global_spectral_avg = np.array([])

            chunk_start = 0
            previous_chunk_size = 0
            coords_df = pd.DataFrame()

            for i in range(num_chunks):
                start_time = time.time()
                chunk_size_temp = chunk_size_base

                if remainder > i:
                    chunk_size_temp += 1
                print(f'I: {i}; chunck_size: {chunk_size_temp}')
                if i != 0:
                    chunk_start += previous_chunk_size
                print(f'chunk_start: {chunk_start}')

                gc.collect()

                if not self.inhomogeneous:
                    chunk_ms_dict = get_chunk_ms_info2(
                        p, chunk_start, chunk_size_temp)
                else:
                    chunk_ms_dict = get_chunk_ms_info_inhomogeneous(
                        p, chunk_start, chunk_size_temp, self.max_mz, self.min_mz, self.mz_RP, self.RP, self.dist, self.rp_factor)
                    chunk_ms_dict['mz'] = self.mzs

                I_mean = np.mean(chunk_ms_dict["I"], axis=0)

                dtype = np.float32

                if peak_alignment:
                    print("aligning . . .")
                    if self.gpu:
                        gc.collect()
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()

                        if align_reduce:
                            reduced_index = [
                                np.arange(val-reduce_halfwidth, val + reduce_halfwidth, 1) for val in self.p2]
                            reduced_index = np.concatenate(reduced_index)
                            reduced_index = reduced_index[reduced_index <
                                                          self.mzs.shape[0]-1]
                            alinger = Aligner_GPU(x=chunk_ms_dict['mz'][reduced_index],
                                                  peaks=self.p2,
                                                  array=chunk_ms_dict["I"][:,
                                                                           reduced_index],
                                                  method="gpu_linear",
                                                  align_by_index=True,
                                                  only_shift=True,
                                                  return_shifts=True,
                                                  width=10,
                                                  ratio=10,
                                                  grid_steps=grid_iter_num,
                                                  shift_range=cp.asarray(
                                                      [-align_halfwidth, align_halfwidth])
                                                  )

                            alinger.run()
                            aligned_I, shifts = alinger.apply()
                            chunk_ms_dict["I"][:,
                                               reduced_index] = aligned_I.get()
                        else:
                            alinger = Aligner_GPU(x=chunk_ms_dict['mz'],
                                                  peaks=self.p2[self.avg_spectrum[self.p2]
                                                                > 100*self.noise],
                                                  array=chunk_ms_dict["I"],
                                                  method="gpu_linear",
                                                  align_by_index=True,
                                                  only_shift=True,
                                                  return_shifts=True,
                                                  width=10,
                                                  ratio=10,
                                                  grid_steps=grid_iter_num,
                                                  shift_range=cp.asarray(
                                                      [-align_halfwidth, align_halfwidth])
                                                  )

                            alinger.run()
                            aligned_I, shifts = alinger.apply()
                            chunk_ms_dict["I"] = aligned_I.get()
                    else:
                        gc.collect()

                        alinger = Aligner_CPU(x=chunk_ms_dict['mz'],
                                              peaks=self.p2,
                                              array=chunk_ms_dict["I"],
                                              method="linear",
                                              align_by_index=True,
                                              only_shift=False,
                                              return_shifts=True,
                                              width=10,
                                              ratio=10,
                                              grid_steps=grid_iter_num,
                                              shift_range=np.asarray(
                                                  [-align_halfwidth, align_halfwidth])
                                              )

                        alinger.run()
                        aligned_I, shifts = alinger.apply()
                        chunk_ms_dict["I"] = aligned_I

                    print(f"Alignment finished for chunk {i}!")

                    del aligned_I

                # chunk 0 settings
                if i == 0:
                    chunk_0_mz_peaks = chunk_ms_dict['mz'][self.p2]
                    chunk_0_mz = chunk_ms_dict['mz']

                    print('"peak_area_df" was not defined.')
                    peaks_df = pd.DataFrame(np.zeros(
                        (len(self.p2), 0)), index=chunk_0_mz_peaks)

                if plot_aligned_peak:
                    plt.style.use("classic")
                    for spectrum_index in range(0, chunk_ms_dict['I'].shape[0], int(chunk_ms_dict['I'].shape[0]/plot_num_peaks)):
                        plt.plot(
                            chunk_ms_dict['I'][spectrum_index][self.p2[index_peak_plot]-20:self.p2[index_peak_plot]+20])
                        plt.plot(
                            20, chunk_ms_dict['I'][spectrum_index][self.p2[index_peak_plot]], "x")
                        plt.hlines(*np.concatenate(([chunk_ms_dict['I'][spectrum_index][self.p2[index_peak_plot]]/2],
                                                    np.asarray(self.p2_width)[2:, index_peak_plot]-self.p2[index_peak_plot]+20),
                                                   axis=0), color="C2")

                        # plot_num_peaks=10
                        # index_peak_plot=23
                        # for spectrum_index in range(0,chunk_ms_dict['I'].shape[0],int(chunk_ms_dict['I'].shape[0]/plot_num_peaks)):
                        #     plt.plot(chunk_ms_dict['I'][spectrum_index][preprocess.p2[index_peak_plot]-20:preprocess.p2[index_peak_plot]+20])
                        #     plt.plot(20, chunk_ms_dict['I'][spectrum_index][preprocess.p2[index_peak_plot]], "x")
                        #     plt.hlines(*np.concatenate(([chunk_ms_dict['I'][spectrum_index][preprocess.p2[index_peak_plot]]/2],
                        #                                 np.asarray(preprocess.p2_width)[2:,index_peak_plot]-preprocess.p2[index_peak_plot]+20),
                        #                                 axis=0), color="C2")

                # Peak integration
                gc.collect()

                if not self.inhomogeneous:
                    peak_area_df_temp = integrate_peak(chunk_ms_dict, self.p2, self.p2_width, chunk_0_mz_peaks,
                                                       method="peak_width", remove_duplicates=False)
                else:
                    # binned MS is automatically centroided
                    peak_area_df_temp = pd.DataFrame(
                        chunk_ms_dict["I"].T[self.p2, :], index=chunk_0_mz_peaks)

                peak_area_df_temp = peak_area_df_temp.rename(
                    columns=lambda s: str(int(s) + chunk_start))
                coords_df_temp = pd.DataFrame(chunk_ms_dict["coords"])
                coords_df_temp = coords_df_temp.rename(
                    index=lambda s: str(int(s) + chunk_start))
                peaks_df = pd.concat([peaks_df, peak_area_df_temp], axis=1)
                coords_df = pd.concat([coords_df, coords_df_temp], axis=0)

                previous_chunk_size = chunk_size_temp
                print(f"Time used for processing chunk {i}: {time.time()-start_time}")
                del peak_area_df_temp, coords_df_temp

            self.peaks_df = peaks_df
            self.coords_df = coords_df

            print("Starting to write to file")
            if os.path.exists(f'{self.data_dir}') == False:
                os.makedirs(f'{self.data_dir}')

            peaks_df.to_csv(
                (f'{self.data_dir}/{dataset[:-6]}.csv'), index=True)
            coords_df.to_csv(
                # 0,1 = x,y
                (f'{self.data_dir}/{dataset[:-6]}_coords.csv'), index=True)

            del peaks_df, coords_df


# test_OOP = Preprocess(directory, data_dir, file_name, lw, up, dist)
# test_OOP.peak_pick()
# test_OOP.run()
# test_OOP.write_file()


# p = ImzMLParser_chunk(directory)


# del peak_area_df, chunk_ms_dict, left_width_list, left_width_mean, right_width_list, right_width_mean
