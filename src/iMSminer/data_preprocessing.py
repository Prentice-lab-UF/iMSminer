# -*- coding: utf-8 -*-
"""
iMSminer beta
@author: Yu Tin Lin (yutinlin@stanford.edu)
@author: Haohui Bao (susanab20020911@gmail.com)
@author: Troy R. Scoggins IV (t.scoggins@ufl.edu)
@author: Boone M. Prentice (booneprentice@ufl.chem.edu)
License: Apache-2.0 
"""

import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
from pyimzml.ImzMLParser import ImzMLParser
from scipy.signal import find_peaks, peak_widths
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .ImzMLParser_chunk import ImzMLParser_chunk
from .utils import (
    Aligner_CPU,
    chunk_prep,
    chunk_prep_inhomogeneous,
    get_chunk_ms_info2,
    get_chunk_ms_info_inhomogeneous,
    get_spectrum,
    integrate_peak,
    str2bool,
)

try:
    import cupy as cp

    from iMSminer.utils import Aligner_GPU
except:
    pass


# ========================DATA PROCESSING========================#


def prompt_for_attributes(prompt_func):
    """
    Decorator for question prompts
    """

    def decorator(method):
        def wrapper(self, *args, **kwargs):
            attributes = method(self, *args, **kwargs)
            for attr_name, prompt in attributes.items():
                # Check if attribute is None or does not exist
                if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
                    setattr(self, attr_name, prompt())
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


def prompt_func(self, attr_name):
    """
    Question prompts for various attributes for class Preprocess()
    """
    prompts = {
        "dist": lambda: int(
            input("Enter the minimum number of data points between near-isobars: ")
        ),
        "loq": lambda: float(
            input(
                "What is your limit of quantification? Enter a coefficient k such that LOQ = k * noise: "
            )
        ),
        "pp_dataset": lambda: input(
            "Enter dataset name (without '') to perform peak picking on: "
        ),
        "lwr": lambda: float(input("Enter a lower m/z bound of noise: ")),
        "upr": lambda: float(input("Enter an upper m/z bound of noise: ")),
        "noise": lambda: float(
            input(
                "Specify a noise level. Enter a number noise such that LOQ = k * noise: "
            )
        ),
        "z_score": lambda: float(
            input("Enter a z_score bound for noise classification: ")
        ),
        "RP": lambda: float(input("Specify FWHF: ")),
        "mz_RP": lambda: float(input("Specify the m/z where FWHF was calculated: ")),
        "rp_factor": lambda: float(input("Scale number of bins by a number: ")),
    }
    return prompts[attr_name]()


class Preprocess:
    """
    Contains functions to import imzML, generate interactive mean mass spectrum, perform peak picking, mass alignment, and peak integration

    Attributes
    ----------
    directory : str
        Directory that contains all imzML files to preprocess
    data_dir : str
        Directory to save preprocessed data
    gpu : bool
        True if gpu-accelerated libraries are imported successfully
    dist : int, user input
        Minimum number of datapoints for peak separation
    loq : float, user input
        Number of times the noise level (k * noise) to define limit of quantification used in peak picking
    pp_dataset : str, user input
        File name of dataset to perform peak picking on
    lwr : float, user input
        Lower m/z bound of a region in spectrum without signals
    upr : float, user input
        Upper m/z bound of a region in spectrum without signals
    noise : float
        Noise level to guide peak picking
    z_score : float, user input
        Statistical upper threshold for noise computation
    RP : float
        Resolving power [FWHM] used to bin spectra
    mz_RP : float
        m/z at which RP is calculated
    rp_factor : float, user input
        Method `binning`, factor to scale number of bins; affects mass resolution
    resolution_progress : str
        Sets resolution for binning if `yes`
    """

    def __init__(self):
        self.directory = input(
            "Enter the directory of imzML files to process: ")
        self.data_dir = input(
            "Enter a directory path for saving preprocessed data: ")

        test_library = ["cupy"]
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
        # Initialize attributes as type None
        self.dist = None
        self.loq = None
        self.pp_dataset = None
        self.lwr = None
        self.upr = None
        self.noise = None
        self.z_score = None
        self.RP = None
        self.mz_RP = None
        self.rp_factor = None
        self.resolution_progress = None

    @prompt_for_attributes(prompt_func)
    def check_attributes(self):
        attributes = {
            "dist": lambda: int(
                input(
                    "Enter the minimum number of data points required to discern closely-spaced peaks: "
                )
            ),
            "loq": lambda: float(
                input(
                    "Specify an intensity threshold for peak picking. Enter a coefficient k such that threshold = k * noise: "
                )
            ),
            "pp_dataset": lambda: (
                input("Enter dataset name (without '') to perform peak picking on: ")
                if self.datasets.shape[0] > 1
                else self.datasets[0]
            ),
            "lwr": lambda: (
                float(input("Enter a lower m/z bound of noise: "))
                if self.pp_method == "point"
                else None
            ),
            "upr": lambda: (
                float(input("Enter an upper m/z bound of noise: "))
                if self.pp_method == "point"
                else None
            ),
            "noise": lambda: (
                float(
                    input(
                        "Specify a noise level. Enter a number noise such that LOQ = k * noise: "
                    )
                )
                if self.pp_method == "specify_noise"
                else None
            ),
            "z_score": lambda: (
                float(
                    input(
                        "Enter a z_score bound (number of standard deviations) for noise classification: "
                    )
                )
                if self.pp_method in ["automatic", "binning_even", "binning_regression"]
                else None
            ),
        }
        if self.pp_method in ["binning_even", "binning_regression"]:
            attributes.update(
                {
                    "RP": lambda: float(input("Specify the resolving power (FWHM): ")),
                    "mz_RP": lambda: float(
                        input(
                            "Specify the m/z at which resolving power was calculated: "
                        )
                    ),
                    "rp_factor": lambda: float(
                        input(
                            "Scale number of m/z bins up or down by a coefficient k (i.e., k * (m/z range of MS analysis) * FWHM / (m/z) = number of bins): "
                        )
                    ),
                }
            )
        return attributes

    def peak_pick(
        self,
        percent_RAM: float = 5,
        pp_method: str = "point",
        rel_height: float = 0.9,
        peak_alignment: bool = False,
        align_threshold: float = 1,
        align_halfwidth: int = 100,
        grid_iter_num: int = 20,
        align_reduce: bool = False,
        reduce_halfwidth: int = 200,
        plot_aligned_peak: bool = True,
        index_peak_plot: int = 0,
        plot_num_peaks: int = 10,
        baseline_subtract: bool = True,
        baseline_method: str = "noise",
    ):
        """Perform peak picking to locate signals above a defined LOQ (k * noise) by specifying k, calculating noise, and at specified minimum distance between peaks

        Parameters
        ----------
        percent_RAM : int, optional
            Percent available RAM occupied by chunk, by default 5
        pp_method : str, optional
            Method of computing noise, by default "point"
            Method `point` takes specified lower and upper m/z bound of a region in spectrum without signals and compute its standard deviation to define noise level
            Method `specify_noise` takes user-specified noise level
            Method `automatic` computes standard deviation based on spectral data points with a  z-score below a threshold k * z-score, where k is specified by user.
            Method `binning` re-bins mass spectra (userful for compressed data with inhomogeneous shapes), then computes noise using method "automatic"
        rel_height : float
            Peak height cutoff for peak integration, by default 0.9
        peak_alignment : bool
            Performs peak alignment if True, by default False. Peak alignment function refactored from (https://github.com/lukasz-migas/msalign)
        align_threshold : float
            Coefficient to define for peaks for alignment, where peaks above align_threshold*noise are aligned
        align_halfwidth : int
            Half width [data points] to define window for mass alignment around a specified peak
        grid_iter_num : int
            Number of steps to be used in the grid search. Default: 20
        align_reduce : bool
            Reduces size of m/z and intensity arrays used in alignment if True, by default False
        reduce_halfwidth: int
            Half width [data points] to define reduction size of m/z and intensity arrays used in alignment if `align_reduce=True`, by default 200
        plot_aligned_peak : bool
            Plots a specified peak after alignment if True, by default True
        index_peak_plot : int
            Index of peak to plot if `plot_aligned_peak=True`, by default 0
        plot_num_peaks : int
            Number of peaks to plot if `plot_aligned_peak=True`, by deault 10
        baseline_subtract : bool
            Calculates baseline and subtracts all intensities from baseline if `baseline_subtract=True`
        baseline_method : str
            Method of baseline calculation if `baseline_subtract=True`
            Method `regression` defines baseline using polynomial regression of input degree
            Method `noise` defines baseline as input coefficient * noise
        """
        self.pp_method = pp_method
        self.percent_RAM = percent_RAM
        self.align_halfwidth = align_halfwidth
        self.grid_iter_num = grid_iter_num
        self.align_reduce = align_reduce
        self.plot_aligned_peak = plot_aligned_peak
        self.align_threshold = align_threshold
        self.index_peak_plot = index_peak_plot
        self.plot_num_peaks = plot_num_peaks
        self.baseline_subtract = baseline_subtract
        self.baseline_method = baseline_method

        datasets = os.listdir(self.directory)
        datasets = np.asarray(datasets)[
            (np.char.find(datasets, "imzML") != -1)]
        self.datasets = datasets
        self.rel_height = rel_height
        self.check_attributes()

        self.peak_alignment = False
        self.peak_pick_func()
        if peak_alignment:
            self.peak_alignment = peak_alignment
            self.peak_pick_func()

    def run(
        self,
        percent_RAM: float = 5,
        peak_alignment: bool = False,
        integrate_method: str = "peak_width",
        align_halfwidth: int = 100,
        grid_iter_num: int = 20,
        align_reduce: bool = True,
        reduce_halfwidth: int = 200,
        plot_aligned_peak: bool = True,
        index_peak_plot: int = 0,
        plot_num_peaks: int = 10,
    ):
        """imports imzML files and perform peak-picking, mass alignment, and peak integration

        Parameters
        ----------
        percent_RAM : int, optional
            Percent available RAM occupied by chunk, by default 5
        peak_alignment : bool, optional, user input
            Performs mass alignment on peaks detected by peak picking if True, by default False
        align_halfwidth : int, user input
            Half-width of window for alignment, by default 100
        grid_iter_num : int, user input
            Number of steps by grid search, by default 20. Larger values give more accurate quantification results but computation time increases quadratically
        align_reduce : bool, optional
            Reduce the size of intensity matrix passed into alignment if True, by default True
        reduce_halfwidth : int, user input
            Half-width of window around peaks for which intensity matrix is reduced before passing into the mass alignment function if True, by default 200
        plot_aligned_peak : bool, optional
            Render a figure to show peak alignment results if True, by default True
        index_peak_plot : int, user input
            Peak with specified analyte index to visualize if plot_aligned_peak, by default 0
        plot_num_peaks : int, user input
            Number of peaks (spectra) at index_peak_plot to plot if True, by default 10
        """
        self.peak_alignment = peak_alignment
        self.align_halfwidth = align_halfwidth
        self.grid_iter_num = grid_iter_num
        self.align_reduce = align_reduce
        self.plot_aligned_peak = plot_aligned_peak
        self.index_peak_plot = index_peak_plot
        self.plot_num_peaks = plot_num_peaks

        for dataset in self.datasets:
            if not self.inhomogeneous:
                p = ImzMLParser_chunk(f"{self.directory}/{dataset}")
                num_chunks, chunk_size_base, chunk_start, remainder = chunk_prep(
                    p, percent_RAM
                )
            else:
                p = ImzMLParser(f"{self.directory}/{dataset}")
                num_chunks, chunk_size_base, chunk_start, remainder = (
                    chunk_prep_inhomogeneous(p, self.n_bins, percent_RAM)
                )
            num_spectra = len(p.coordinates)
            chunk_start = 0
            previous_chunk_size = 0
            coords_df = pd.DataFrame()

            for i in range(num_chunks):
                start_time = time.time()
                chunk_size_temp = chunk_size_base

                if remainder > i:
                    chunk_size_temp += 1
                print(f"chunk: {i+1} of {num_chunks}; \n"
                      f"chunck_size: {chunk_size_temp} of {num_spectra}")
                if i != 0:
                    chunk_start += previous_chunk_size

                if not self.inhomogeneous:
                    chunk_ms_dict = get_chunk_ms_info2(
                        p, chunk_start, chunk_size_temp)
                else:
                    chunk_ms_dict = get_chunk_ms_info_inhomogeneous(
                        p,
                        chunk_start,
                        chunk_size_temp,
                        self.max_mz,
                        self.min_mz,
                        self.mz_RP,
                        self.RP,
                        self.dist,
                        self.rp_factor,
                    )
                    chunk_ms_dict["mz"] = self.mzs
                if self.baseline_subtract:
                    chunk_ms_dict["I"] -= self.baseline_pred
                    chunk_ms_dict["I"][chunk_ms_dict["I"] < 0] = 0

                chunk_ms_dict["mz"], chunk_ms_dict["I"] = self.peak_alignment_func(
                    mz=chunk_ms_dict["mz"], intensity=chunk_ms_dict["I"]
                )

                # chunk 0 settings
                if i == 0:
                    peaks_df = pd.DataFrame(
                        np.zeros((len(self.p2), 0)), index=chunk_ms_dict["mz"][self.p2]
                    )

                if not self.inhomogeneous:
                    peak_area_df_temp = integrate_peak(
                        chunk_ms_dict,
                        self.p2,
                        self.p2_width,
                        self.dist,
                        chunk_ms_dict["mz"][self.p2],
                        integrate_method=integrate_method,
                    )
                else:
                    peak_area_df_temp = pd.DataFrame(
                        chunk_ms_dict["I"].T[self.p2, :],
                        index=chunk_ms_dict["mz"][self.p2],
                    )

                peak_area_df_temp = peak_area_df_temp.rename(
                    columns=lambda s: str(int(s) + chunk_start)
                )
                coords_df_temp = pd.DataFrame(chunk_ms_dict["coords"])
                coords_df_temp = coords_df_temp.rename(
                    index=lambda s: str(int(s) + chunk_start)
                )
                peaks_df = pd.concat([peaks_df, peak_area_df_temp], axis=1)
                coords_df = pd.concat([coords_df, coords_df_temp], axis=0)

                previous_chunk_size = chunk_size_temp
                print(
                    f"Time used for processing chunk {i}: {time.time()-start_time}")
                del peak_area_df_temp, coords_df_temp
            gc.collect()
            self.peaks_df = peaks_df
            self.coords_df = coords_df

            print("Starting to write to file")
            if os.path.exists(f"{self.data_dir}") == False:
                os.makedirs(f"{self.data_dir}")

            peaks_df.to_csv(
                (f"{self.data_dir}/{dataset[:-6]}.csv"), index=True)
            coords_df.to_csv(
                # 0,1 = x,y
                (f"{self.data_dir}/{dataset[:-6]}_coords.csv"),
                index=True,
            )

            del peaks_df, coords_df

    def peak_pick_func(self):
        """Performs peak picking using method `point`, `specify_noise`, `automatic`, `binning_even` or `binning_regression`, with optional baseline subtraction using method `regression` or `noise`"""

        try:
            self.p = ImzMLParser_chunk(f"{self.directory}/{self.pp_dataset}")
            self.num_chunks, self.chunk_size_base, self.chunk_start, self.remainder = (
                chunk_prep(self.p, self.percent_RAM)
            )
            if self.pp_method == "point":
                p2, p2_width = self.get_p2(method=self.pp_method)
            elif self.pp_method == "specify_noise":
                p2, p2_width = self.get_p2(method=self.pp_method)
            elif self.pp_method == "automatic":
                index_stop = int(self.p.getmz(0).shape[0] * 0.999)
                avg_spectrum = np.zeros(index_stop)
                mzs = self.p.getmz(0)[:index_stop]
                self.mzs = mzs
                noise_array = np.zeros(index_stop)
                for i in range(self.num_chunks):
                    print(f"Chunk {i}")
                    start_time = time.time()
                    chunk_size_temp = self.chunk_size_base
                    if self.remainder > i:
                        chunk_size_temp += 1
                    if i != 0:
                        self.chunk_start += chunk_size_temp

                    intensities_chunk = self.p.get_intensity_chunk(
                        chunk_start=self.chunk_start,
                        chunk_end=(self.chunk_start + chunk_size_temp),
                        index_stop=index_stop,
                    )
                    mzs, intensities_chunk = self.peak_alignment_func(
                        mz=mzs, intensity=intensities_chunk
                    )

                    try:
                        if self.plot_aligned_peak:
                            plt.style.use("classic")
                            for spectrum_index in range(
                                0,
                                intensities_chunk.shape[0],
                                int(intensities_chunk.shape[0] /
                                    self.plot_num_peaks),
                            ):
                                plt.plot(
                                    intensities_chunk[spectrum_index][
                                        self.p2[self.index_peak_plot]
                                        - 20: self.p2[self.index_peak_plot]
                                        + 20
                                    ]
                                )
                                plt.plot(
                                    20,
                                    intensities_chunk[spectrum_index][
                                        self.p2[self.index_peak_plot]
                                    ],
                                    "x",
                                )
                                plt.hlines(
                                    *np.concatenate(
                                        (
                                            [
                                                intensities_chunk[spectrum_index][
                                                    self.p2[self.index_peak_plot]
                                                ]
                                                / 2
                                            ],
                                            np.asarray(self.p2_width)[
                                                2:, self.index_peak_plot
                                            ]
                                            - self.p2[self.index_peak_plot]
                                            + 20,
                                        ),
                                        axis=0,
                                    ),
                                    color="C2",
                                )
                            plt.show()
                    except AttributeError:
                        pass

                    avg_spectrum += np.sum(intensities_chunk, axis=0)
                    noise_chunk = np.std(intensities_chunk, axis=0)
                    noise_array += noise_chunk

                    print(
                        f"Time used for running chunk {i+1} with a size of \n"
                        f"{chunk_size_temp}: {time.time()-start_time}"
                    )
                gc.collect()
                avg_spectrum /= len(self.p.coordinates)
                self.avg_spectrum = avg_spectrum

                noise = (
                    np.mean(
                        noise_array[abs(scipy.stats.zscore(noise_array)) < 3])
                    / self.num_chunks
                )
                self.noise = noise
                self.noise_array = noise_array / self.num_chunks
                self.baseline_subtraction()

                p2 = find_peaks(
                    x=self.avg_spectrum, height=self.loq * noise, distance=self.dist
                )
                p2 = p2[0]
                p2_width = peak_widths(
                    self.avg_spectrum, p2, rel_height=self.rel_height
                )
                p2_width = np.asarray(p2_width)

            self.inhomogeneous = False
            self.p2 = p2
            self.p2_width = p2_width

            get_spectrum(
                output_filepath=f"{self.data_dir}/\n"
                f"{self.pp_dataset[:-6]}_avg_spectrum.html",
                mzs=self.mzs,
                avg_intensity=self.avg_spectrum,
                p2=self.p2,
            )

        except Exception:
            self.baseline_subtract = False
            if self.pp_method not in ["binning_even", "binning_regression"]:
                self.pp_method = input(
                    "Select binning method: `binning_even` or `binning_regression`. Use `binning_regression`, which requires simple linear regression equation from original data, for more optimal mass resolution. "
                )
            self.check_attributes()
            self.p = ImzMLParser(f"{self.directory}/{self.pp_dataset}")
            self.num_chunks, self.chunk_size_base, self.chunk_start, self.remainder = (
                chunk_prep(self.p, self.percent_RAM)
            )
            min_mz = 10**10
            max_mz = 0
            for i in range(len(self.p.coordinates)):
                mz = self.p.getspectrum(i)[0]
                if min(mz) < min_mz:
                    min_mz = min(mz)
                if max(mz) > max_mz:
                    max_mz = max(mz)

            self.min_mz = min_mz
            self.max_mz = max_mz
            if self.pp_method == "binning_even":
                resolution_exit = False
                while not resolution_exit:
                    n_bins = (
                        int((max_mz - min_mz) /
                            (self.mz_RP / self.RP) * self.rp_factor)
                        + 1
                    )
                    self.n_bins = n_bins
                    avg_spectrum = np.zeros(n_bins)
                    self.mzs = np.linspace(min_mz, max_mz, num=n_bins)

                    num_chunks, chunk_size_base, chunk_start, remainder = (
                        chunk_prep_inhomogeneous(
                            self.p, n_bins, self.percent_RAM)
                    )
                    previous_chunk_size = 0
                    num_spectra = len(self.p.coordinates)
                    for i in range(num_chunks):
                        chunk_size_temp = chunk_size_base

                        if remainder > i:
                            chunk_size_temp += 1

                        if i != 0:
                            chunk_start += previous_chunk_size
                        print(f"chunk: {i+1} of {num_chunks}; \n"
                              f"chunck_size: {chunk_size_temp} of {num_spectra}")

                        chunk_ms_dict = get_chunk_ms_info_inhomogeneous(
                            self.p,
                            chunk_start,
                            chunk_size_temp,
                            self.max_mz,
                            self.min_mz,
                            self.mz_RP,
                            self.RP,
                            self.dist,
                            self.rp_factor,
                        )
                        self.mzs, chunk_ms_dict["I"] = self.peak_alignment_func(
                            mz=self.mzs, intensity=chunk_ms_dict["I"]
                        )
                        avg_spectrum += np.sum(chunk_ms_dict["I"], axis=0)
                        previous_chunk_size = chunk_size_temp

                    avg_spectrum /= len(self.p.coordinates)

                    self.noise = np.std(
                        avg_spectrum[
                            abs(scipy.stats.zscore(avg_spectrum)) < self.z_score
                        ]
                    )

                    p2 = find_peaks(
                        x=avg_spectrum, height=self.loq * self.noise, distance=self.dist
                    )
                    p2_width = peak_widths(
                        avg_spectrum, p2[0], rel_height=self.rel_height
                    )

                    self.p2 = p2[0]
                    self.p2_width = np.asarray(p2_width)
                    self.avg_spectrum = avg_spectrum
                    self.inhomogeneous = True

                    get_spectrum(
                        output_filepath=f"{self.data_dir}/\n"
                        f"{self.pp_dataset[:-6]}_avg_spectrum.html",
                        mzs=self.mzs,
                        avg_intensity=avg_spectrum,
                        p2=self.p2,
                    )

                    resolution_progress = input(
                        "Satisfied with resolution? (yes/no) ")
                    if resolution_progress == "yes":
                        resolution_exit = True
                    else:
                        resolution_exit = False
                        self.rp_factor = float(
                            input("Scale number of bins by a number: ")
                        )
                        self.dist = int(
                            input(
                                "What is the minimum number of data points between peaks? "
                            )
                        )

            elif self.pp_method == "binning_regression":
                self.inhomogeneous = True
                try:
                    self.regression_eq
                except AttributeError:
                    ols_exit = False
                    while not ols_exit:
                        regression_eq = input(
                            "Specify a simple linear regression equation of intercept + coefficient. Enter: intercept{one space}coefficient "
                        )
                        regression_eq = regression_eq.split(" ")
                        if len(regression_eq) != 2:
                            print(
                                "Must enter a simple linear regression equation with an intercept and a coefficient for slope. ")
                            ols_exit = False
                        else:
                            ols_exit = True

                    regression_eq = np.asarray(
                        regression_eq).astype(np.float32)
                    self.regression_eq = regression_eq

                resolution_exit = False
                while not resolution_exit:
                    n_bins = (
                        int((max_mz - min_mz) /
                            (self.mz_RP / self.RP) * self.rp_factor)
                        + 1
                    )
                    self.n_bins = n_bins
                    avg_spectrum = np.zeros(n_bins)
                    self.mzs = np.linspace(min_mz, max_mz, num=n_bins)

                    num_chunks, chunk_size_base, chunk_start, remainder = (
                        chunk_prep_inhomogeneous(
                            self.p, n_bins, self.percent_RAM)
                    )
                    previous_chunk_size = 0
                    num_spectra = len(self.p.coordinates)
                    for i in range(num_chunks):
                        chunk_size_temp = chunk_size_base

                        if remainder > i:
                            chunk_size_temp += 1

                        if i != 0:
                            chunk_start += previous_chunk_size
                        print(f"chunk: {i+1} of {num_chunks}; \n"
                              f"chunck_size: {chunk_size_temp} of {num_spectra}")

                        chunk_ms_dict = get_chunk_ms_info_inhomogeneous(
                            self.p,
                            chunk_start,
                            chunk_size_temp,
                            self.max_mz,
                            self.min_mz,
                            self.mz_RP,
                            self.RP,
                            self.dist,
                            self.rp_factor,
                        )

                        avg_spectrum += np.sum(chunk_ms_dict["I"], axis=0)
                        previous_chunk_size = chunk_size_temp

                    avg_spectrum /= len(self.p.coordinates)

                    self.noise = np.std(
                        avg_spectrum[
                            abs(scipy.stats.zscore(avg_spectrum)) < self.z_score
                        ]
                    )

                    p2 = find_peaks(
                        x=avg_spectrum, height=self.loq * self.noise, distance=self.dist
                    )
                    self.p2 = p2[0]
                    p2_half_width = peak_widths(
                        avg_spectrum, p2[0], rel_height=0.5)
                    p2_half_width = np.asarray(p2_half_width)

                    p2_half_width = p2_half_width[2:].T
                    scale_factor = np.ceil(p2_half_width) - p2_half_width

                    self.FWHM = [
                        self.mzs[self.p2[i]]
                        / (
                            self.mzs[np.floor(row[1]).astype(int)]
                            * (1 - scale_factor[i][1])
                            + self.mzs[np.ceil(row[1]).astype(int)]
                            * (scale_factor[i][1])
                            - (
                                self.mzs[np.floor(row[0]).astype(int)]
                                * (1 - scale_factor[i][0])
                                + self.mzs[np.ceil(row[0]).astype(int)]
                                * (scale_factor[i][0])
                            )
                        )
                        for i, row in enumerate(p2_half_width)
                    ]

                    avg_spectrum = np.zeros(n_bins)

                    rp_predicted = self.regression_eq[0] + self.regression_eq[
                        1
                    ] * np.linspace(min_mz, max_mz, n_bins)
                    rp_predicted = rp_predicted / np.sum(rp_predicted)

                    self.mzs = np.interp(
                        np.linspace(0, 1, n_bins),
                        np.cumsum(rp_predicted),
                        np.linspace(min_mz, max_mz, n_bins),
                    )

                    previous_chunk_size = 0
                    chunk_start = 0

                    for i in range(num_chunks):
                        chunk_size_temp = chunk_size_base

                        if remainder > i:
                            chunk_size_temp += 1

                        if i != 0:
                            chunk_start += previous_chunk_size
                        print(f"chunk: {i+1} of {num_chunks}; \n"
                              f"chunck_size: {chunk_size_temp} of {num_spectra}")

                        chunk_ms_dict = get_chunk_ms_info_inhomogeneous(
                            self.p,
                            chunk_start,
                            chunk_size_temp,
                            self.max_mz,
                            self.min_mz,
                            self.mz_RP,
                            self.RP,
                            self.dist,
                            self.rp_factor,
                        )
                        self.mzs, chunk_ms_dict["I"] = self.peak_alignment_func(
                            mz=self.mzs, intensity=chunk_ms_dict["I"]
                        )
                        avg_spectrum += np.sum(chunk_ms_dict["I"], axis=0)
                        previous_chunk_size = chunk_size_temp

                    avg_spectrum /= len(self.p.coordinates)

                    self.noise = np.std(
                        avg_spectrum[
                            abs(scipy.stats.zscore(avg_spectrum)) < self.z_score
                        ]
                    )

                    p2 = find_peaks(
                        x=avg_spectrum, height=self.loq * self.noise, distance=self.dist
                    )
                    p2_width = peak_widths(
                        avg_spectrum, p2[0], rel_height=self.rel_height
                    )

                    self.p2 = p2[0]
                    self.p2_width = np.asarray(p2_width)
                    self.avg_spectrum = avg_spectrum
                    self.inhomogeneous = True

                    get_spectrum(
                        output_filepath=f"{self.data_dir}/\n"
                        f"{self.pp_dataset[:-6]}_avg_spectrum.html",
                        mzs=self.mzs,
                        avg_intensity=avg_spectrum,
                        p2=self.p2,
                    )

                    resolution_progress = input(
                        "Satisfied with resolution? (yes/no) ")
                    if resolution_progress == "yes":
                        resolution_exit = True
                    else:
                        resolution_exit = False
                        self.rp_factor = float(
                            input("Scale number of bins by a number: ")
                        )
                        self.dist = int(
                            input(
                                "What is the minimum number of data points between peaks? "
                            )
                        )

            else:
                print("Data has inhomogeneous shape. Default to binning.\
                      Mass alignment is recommended for optimal binning. ")
                print(
                    "Unrecognized method. Choose an option from `automatic`, `specify_noise`, or `point` if raw imzML, and `binning_even` or `binning_regression` if reduced imzML."
                )
                return

    def peak_alignment_func(self, mz, intensity):
        """ALigns input intensity array based on peak index or m/z values

        Parameters
        ----------
        mz : np.ndarray
            m/z array used in alignment
        intensity : np.ndarray
            Intensity array before alignment

        Returns
        -------
        mz : np.ndarray
            m/z array used in alignment
        intensity : np.ndarray
            Aligned intesnsity array

        """
        if self.peak_alignment:
            print("aligning . . .")
            if self.gpu:
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

                if self.align_reduce:
                    reduced_index = [
                        np.arange(
                            val - self.reduce_halfwidth, val + self.reduce_halfwidth, 1
                        )
                        for val in self.p2
                    ]
                    reduced_index = np.concatenate(reduced_index)
                    reduced_index = reduced_index[reduced_index <
                                                  self.mzs.shape[0] - 1]
                    alinger = Aligner_GPU(
                        x=mz[reduced_index],
                        peaks=mz[self.p2][
                            self.avg_spectrum[self.p2]
                            > self.align_threshold * self.noise
                        ],
                        array=intensity[:, reduced_index],
                        method="gpu_linear",
                        align_by_index=True,
                        only_shift=True,
                        return_shifts=True,
                        width=10,
                        ratio=10,
                        grid_steps=self.grid_iter_num,
                        shift_range=cp.asarray(
                            [-self.align_halfwidth, self.align_halfwidth]
                        ),
                    )

                    alinger.run()
                    aligned_I, shifts = alinger.apply()
                    intensity[:, reduced_index] = aligned_I.get()
                else:
                    alinger = Aligner_GPU(
                        x=mz,
                        peaks=mz[self.p2][
                            self.avg_spectrum[self.p2]
                            > self.align_threshold * self.noise
                        ],
                        array=intensity,
                        method="gpu_linear",
                        align_by_index=True,
                        only_shift=True,
                        return_shifts=True,
                        width=10,
                        ratio=10,
                        grid_steps=self.grid_iter_num,
                        shift_range=cp.asarray(
                            [-self.align_halfwidth, self.align_halfwidth]
                        ),
                    )

                    alinger.run()
                    aligned_I, shifts = alinger.apply()
                    intensity = aligned_I.get()
            else:
                if self.align_reduce:
                    reduced_index = [
                        np.arange(
                            val - self.reduce_halfwidth, val + self.reduce_halfwidth, 1
                        )
                        for val in self.p2
                    ]
                    reduced_index = np.concatenate(reduced_index)
                    reduced_index = reduced_index[reduced_index <
                                                  self.mzs.shape[0] - 1]
                    alinger = Aligner_CPU(
                        x=mz[reduced_index],
                        peaks=mz[self.p2][
                            self.avg_spectrum[self.p2]
                            > self.align_threshold * self.noise
                        ],
                        array=intensity[:, reduced_index],
                        method="linear",
                        align_by_index=True,
                        only_shift=True,
                        return_shifts=True,
                        width=10,
                        ratio=10,
                        grid_steps=self.grid_iter_num,
                        shift_range=cp.asarray(
                            [-self.align_halfwidth, self.align_halfwidth]
                        ),
                    )

                    alinger.run()
                    aligned_I, shifts = alinger.apply()
                    intensity[:, reduced_index] = aligned_I
                else:
                    gc.collect()

                    alinger = Aligner_CPU(
                        x=mz,
                        peaks=mz[self.p2][
                            self.avg_spectrum[self.p2]
                            > self.align_threshold * self.noise
                        ],
                        array=intensity,
                        method="linear",
                        align_by_index=True,
                        only_shift=False,
                        return_shifts=True,
                        width=10,
                        ratio=10,
                        grid_steps=self.grid_iter_num,
                        shift_range=np.asarray(
                            [-self.align_halfwidth, self.align_halfwidth]
                        ),
                    )

                    alinger.run()
                    aligned_I, shifts = alinger.apply()
                    intensity = aligned_I

            del aligned_I

        return mz, intensity

    def get_p2(self):
        """Calculates noise level on average spectrum, performs peak picking, and calculates peak widths for peak integration

        Returns
        -------
        p2 : np.1darray
            m/z bin indices corresponding to peak-picked maxima
        p2_width : np.1darray
            Peak regions computed by np.peak_widths
        """
        index_stop = int(self.p.getmz(0).shape[0] * 0.999)
        avg_spectrum = np.zeros(index_stop)
        mzs = self.p.getmz(0)[:index_stop]
        noise = 0
        for i in range(self.num_chunks):
            print(f"Chunk {i}")
            start_time = time.time()
            chunk_size_temp = self.chunk_size_base
            if self.remainder > i:
                chunk_size_temp += 1

            intensities_chunk = self.p.get_intensity_chunk(
                chunk_start=self.chunk_start,
                chunk_end=(self.chunk_start + chunk_size_temp),
                index_stop=index_stop,
            )

            mzs, intensities_chunk = self.peak_alignment_func(
                mz=mzs, intensity=intensities_chunk
            )

            avg_spectrum += np.sum(intensities_chunk, axis=0)

            if self.pp_method == "point":
                noise_bin = np.std(
                    np.asarray(intensities_chunk)[
                        :, np.logical_and(mzs >= self.lwr, mzs <= self.upr)
                    ],
                    axis=0,
                )
                if np.sum(noise_bin) == 0:
                    pass
                else:
                    noise += np.mean(noise_bin[noise_bin != 0])
                    if i == (self.num_chunks - 1):
                        self.noise /= self.num_chunks

                self.chunk_start += chunk_size_temp

            print(
                f"Time used for running chunk {i} with a size of \n"
                f"{chunk_size_temp}: {time.time()-start_time}"
            )
            gc.collect()

        avg_spectrum /= len(self.p.coordinates)
        self.avg_spectrum = avg_spectrum
        self.mzs = mzs
        p2, info = find_peaks(
            x=avg_spectrum, height=self.loq * self.noise, distance=self.dist
        )
        p2_width = peak_widths(avg_spectrum, p2, rel_height=self.rel_height)
        p2_width = np.asarray(p2_width)

        return p2, p2_width

    def baseline_subtraction(self):
        try:
            self.baseline_pred
        except AttributeError:
            if self.baseline_subtract:
                if self.baseline_method == "regression":
                    baseline_exit = False
                    while not baseline_exit:
                        baseline_index = abs(
                            scipy.stats.zscore(self.noise_array)) < 2
                        degree = int(
                            input(
                                "Enter degree of regression for baseline computation: "
                            )
                        )
                        poly = PolynomialFeatures(
                            degree=degree, include_bias=False)
                        poly_features = poly.fit_transform(
                            self.mzs[baseline_index].reshape(-1, 1)
                        )
                        poly_reg_model = LinearRegression()
                        poly_reg_model.fit(
                            poly_features, self.noise_array[baseline_index]
                        )
                        R2 = poly_reg_model.score(
                            poly_features, self.noise_array[baseline_index]
                        )

                        poly_features = poly.fit_transform(
                            self.mzs.reshape(-1, 1))
                        baseline_pred = poly_reg_model.predict(poly_features)

                        fig, ax = plt.subplots(figsize=(30, 10))
                        ax.scatter(self.mzs, baseline_pred, color="red")
                        plt.plot(
                            self.mzs[baseline_index],
                            self.noise_array[baseline_index],
                            alpha=0.3,
                            color="black",
                        )
                        ax.set_title(f"Linear model of degree {degree}")
                        ax.set_xlabel("m/z")
                        ax.set_ylabel("Intensity")
                        ax.text(
                            0.5,
                            0.95,
                            f"$R^2 = {R2:.3f}$",
                            fontsize=12,
                            horizontalalignment="center",
                            verticalalignment="top",
                            transform=plt.gca().transAxes,
                        )
                        plt.show()

                        baseline_exit = input(
                            "Satisfied with baseline subtraction? Enter (yes / no): "
                        )
                        baseline_exit = str2bool(baseline_exit)
                    self.baseline_pred = baseline_pred
                elif self.baseline_method == "noise":
                    baseline_factor = float(
                        input(
                            "Enter a coefficient k for baseline, such that baseline = k * noise: "
                        )
                    )
                    self.baseline_pred = self.noise * baseline_factor
                self.avg_spectrum -= self.baseline_pred
                self.avg_spectrum[self.avg_spectrum < 0] = 0
