# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:21:37 2024

@author: yutin
"""



import warnings
import logging
import typing as ty
import cupy as cp
import time

import os

os.chdir("Z:\\Lin\\2023\\Python\\Scripts\\msalign")
from utilities import check_xy, convert_peak_values_to_index, convert_peak_values_to_index_gpu, generate_function, shift, time_loop, find_nearest_index, find_nearest_index_gpu
from nan_to_num_njit_cupy import fillna_cpwhere_njit

import time
import gc

#================Load MS Data=============#
import os
os.chdir("Z:\\Lin\\2023\\Python\\Scripts")
from IMS_Processing_Functions import get_ms_info, lwr_upr_mz, get_int, int_at_mz, normalize, SCiLS_raw_intensities
#from sim_modules_vector import calculate_counters, quick_rr, calculate_comp_sim_rr

import pandas as pd
import numpy as np
from numpy import inf,nan
import multiprocessing as mp
import numexpr as ne
from bokeh.plotting import figure, show, save, output_file

analyte = input("What is your analyte of interest? ")
directory = input("What is your SCiLS dataset directory? ") #directory = Z:\Lin\2023\SCiLS\BTHS\BTHS_CL
data_dir = input("Where to save the spectral data? ") #data_dir = Z:\Lin\2023\Python\Data\BTHS\CL_spectral_matrix
lock_mass = True
generate_spectrum = False
calibration = False

if not os.path.exists(data_dir):
        os.makedirs(data_dir)

if 'global_spectral_avg' in locals(): 
    del global_spectral_avg
    
nf = '885cal'
for file_name in os.listdir(directory):
    if file_name.endswith(".sbd"):       
        print(file_name)

        norI_dict = dict()
        ms_dict = get_ms_info(directory, file_name, nf)
        del ms_dict['coords'] 
        #[ms_dict.get(key)[ms_dict['mz'] <= 1000] for key in ['mz', 'I']]
        #ms_dict['I'] = ms_dict.get('I')[:,ms_dict['mz'] <= 1000]
        #ms_dict['mz'] = ms_dict.get('mz')[ms_dict['mz'] <= 1000]
  
        # if not 'MZ' in locals():
        #     ms_dict['mz'] = ms_dict.get('mz')[ms_dict['mz'] <= 1000]
        #     MZ = ms_dict["mz"]      
        try:
            global_spectral_avg  # Attempt to access the variable
            # Object is defined
            pass
        except NameError:
            # Object is undefined
            global_spectral_avg = np.empty((len(ms_dict['mz']), 0), dtype=object)
            #global_mz = np.empty((len(MZ), 0), dtype=object)
        global_spectral_avg = np.insert(global_spectral_avg,0,np.mean(ms_dict["I"], axis = 0),axis=1)
        #del ms_dict

global_spectral_avg_mean = np.mean(global_spectral_avg, axis=1)


#===============Peak Picking====================#
###peak picking
import seaborn as sns
import matplotlib.pyplot as plt

file_name= list(filter(lambda x: ".sbd" in x, os.listdir(directory)))[0]

norI_dict = dict()
ms_dict = get_ms_info(directory, file_name, nf)


if analyte == "CL":
    lwr, upr = ms_dict['mz'] >= 630, ms_dict['mz'] <= 650 #CL
elif analyte == "FA":
    lwr, upr = ms_dict['mz'] >= 1660, ms_dict['mz'] <= 1700 #FA
elif analyte == "metabolite":
    lwr, upr = ms_dict['mz'] >= 492, ms_dict['mz'] <= 495 #metabolite

noise = np.mean(ms_dict["I"][:,lwr & upr].std(axis=0))  #DO NOT USE I_mean. Watch low noise level, artifact of indexed row names.

select_I = pd.DataFrame({'mz': ms_dict['mz'], 'intensity': global_spectral_avg_mean}, columns=['mz', 'intensity']).set_index('mz') #set index to display mz instead of indices of intensity

######METHOD 2
from scipy.signal import find_peaks, peak_widths
from scipy import stats

select_I.reset_index(inplace=True)

#noise = df.std(axis = 0)[0]
#lwr, upr = select_I.mz >= 620, select_I.mz <= 670 #CL
#np.loadtxt(filename)

h = 10*noise

#prom = 10000
dist = 100
p2, info = find_peaks(x=global_spectral_avg_mean,
height=h,
# prominence=prom,
 distance=dist
 )

    

METHODS = ["pchip", "zero", "slinear", "quadratic", "cubic", "linear", "gpu_linear"]
LOGGER = logging.getLogger(__name__)



class Aligner_CPU:
    """Main alignment class"""

    _method, _gaussian_ratio, _gaussian_resolution, _gaussian_width, _n_iterations = None, None, None, None, None
    _corr_sig_l, _corr_sig_x, _corr_sig_y, _reduce_range_factor, _scale_range = None, None, None, None, None
    _search_space, _computed = None, False

    def __init__(
        self,
        x: np.ndarray,
        array: ty.Optional[np.ndarray],
        peaks: ty.Iterable[float],
        method: str = "cubic",
        width: float = 10,
        ratio: float = 2.5,
        resolution: int = 100,
        iterations: int = 5,
        grid_steps: int = 20,
        shift_range: ty.Optional[ty.Tuple[int, int]] = None,
        weights: ty.Optional[ty.List[float]] = None,
        return_shifts: bool = False,
        align_by_index: bool = False,
        only_shift: bool = False,
    ):
        """Signal calibration and alignment by reference peaks

        A simplified version of the MSALIGN function found in MATLAB (see references for link)

        This version of the msalign function accepts most of the parameters that MATLAB's function accepts with the
        following exceptions: GroupValue, ShowPlotValue. A number of other parameters is allowed, although they have
        been renamed to comply with PEP8 conventions. The Python version is 8-60 times slower than the MATLAB
        implementation, which is mostly caused by a really slow instantiation of the
        `scipy.interpolate.PchipInterpolator` interpolator. In order to speed things up, I've also included several
        other interpolation methods which are significantly faster and give similar results.

        References
        ----------
        Monchamp, P., Andrade-Cetto, L., Zhang, J.Y., and Henson, R. (2007) Signal Processing Methods for Mass
        Spectrometry. In Systems Bioinformatics: An Engineering Case-Based Approach, G. Alterovitz and M.F. Ramoni, eds.
        Artech House Publishers).
        MSALIGN: https://nl.mathworks.com/help/bioinfo/ref/msalign.html

        Parameters
        ----------
        x : np.ndarray
            1D array of separation units (N). The number of elements of xvals must equal the number of elements of
            zvals.shape[1]
        array : np.ndarray
            2D array of intensities that must have common separation units (M x N) where M is the number of vectors
            and N is number of points in the vector
        peaks : list
            list of reference peaks that must be found in the xvals vector
        method : str
            interpolation method. Default: 'cubic'. MATLAB version uses 'pchip' which is significantly slower in Python
        weights: list (optional)
            list of weights associated with the list of peaks. Must be the same length as list of peaks
        width : float (optional)
            width of the gaussian peak in separation units. Default: 10
        ratio : float (optional)
            scaling value that determines the size of the window around every alignment peak. The synthetic signal is
            compared to the input signal within these regions. Default: 2.5
        resolution : int (optional)
            Default: 100
        iterations : int (optional)
            number of iterations. Increasing this value will (slightly) slow down the function but will improve
            performance. Default: 5
        grid_steps : int (optional)
            number of steps to be used in the grid search. Default: 20
        shift_range : list / numpy array (optional)
            maximum allowed shifts. Default: [-100, 100]
        only_shift : bool
            determines if signal should be shifted (True) or rescaled (False). Default: True
        return_shifts : bool
            decide whether shift parameter `shift_opt` should also be returned. Default: False
        align_by_index : bool
            decide whether alignment should be done based on index rather than `xvals` array. Default: False
        """
        self.x = np.asarray(x)
        if array is not None:
            self.array = check_xy(self.x, np.asarray(array))
        else:
            self.array = np.empty((0, len(self.x)))

        self.n_signals = self.array.shape[0]
        self.array_aligned = np.zeros_like(self.array)
        self.peaks = list(peaks)

        # set attributes
        self.n_peaks = len(self.peaks)

        # accessible attributes
        self.scale_opt = np.ones((self.n_signals, 1), dtype=np.float32)
        self.shift_opt = np.zeros((self.n_signals, 1), dtype=np.float32)
        self.shift_values = np.zeros_like(self.shift_opt)

        self.method = method
        self.gaussian_ratio = ratio
        self.gaussian_resolution = resolution
        self.gaussian_width = width
        self.n_iterations = iterations
        self.grid_steps = grid_steps
        if shift_range is None:
            shift_range = [-100, 100]
        self.shift_range = shift_range
        if weights is None:
            weights = np.ones(self.n_peaks)
        self.weights = weights

        # return shift vector
        self._return_shifts = return_shifts
        # If the number of points is equal to 1, then only shift
        if self.n_peaks == 1:
            only_shift = True
        if only_shift and not align_by_index:
            align_by_index = True
            LOGGER.warning("Only computing shifts - changed `align_by_index` to `True`.")

        # align signals by index rather than peak value
        self._align_by_index = align_by_index
        # align by index - rather than aligning to arbitrary non-integer values in the xvals, you can instead
        # use index of those values
        if self._align_by_index:
            self.peaks = convert_peak_values_to_index(self.x, self.peaks)
            self.x = np.arange(self.x.shape[0])
            LOGGER.debug(f"Aligning by index - peak positions: {self.peaks}")
        self._only_shift = only_shift
        

        self._initialize()

    @property
    def method(self):
        """Interpolation method."""
        return self._method

    @method.setter
    def method(self, value: str):
        if value not in METHODS:
            raise ValueError(f"Method `{value}` not found in the method options: {METHODS}")
        self._method = value

    @property
    def gaussian_ratio(self):
        """Gaussian ratio."""
        return self._gaussian_ratio

    @gaussian_ratio.setter
    def gaussian_ratio(self, value: float):
        if value <= 0:
            raise ValueError("Value of 'ratio' must be above 0!")
        self._gaussian_ratio = value

    @property
    def gaussian_resolution(self):
        """Gaussian resolution of every Gaussian pulse (number of points)."""
        return self._gaussian_resolution

    @gaussian_resolution.setter
    def gaussian_resolution(self, value: float):
        if value <= 0:
            raise ValueError("Value of 'resolution' must be above 0!")
        self._gaussian_resolution = value

    @property
    def gaussian_width(self):
        """Width of the Gaussian pulse in std dev of the Gaussian pulses (in X)."""
        return self._gaussian_width

    @gaussian_width.setter
    def gaussian_width(self, value: float):
        self._gaussian_width = value

    @property
    def n_iterations(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, value: int):
        if value < 1 or not isinstance(value, int):
            raise ValueError("Value of 'iterations' must be above 0 and be an integer!")
        self._n_iterations = value

    @property
    def grid_steps(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._grid_steps

    @grid_steps.setter
    def grid_steps(self, value: int):
        if value < 1 or not isinstance(value, int):
            raise ValueError("Value of 'iterations' must be above 0 and be an integer!")
        self._grid_steps = value

    @property
    def shift_range(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._shift_range

    @shift_range.setter
    def shift_range(self, value: ty.Tuple[float, float]):
        if len(value) != 2:
            raise ValueError(
                "Number of 'shift_values' is not correct. Shift range accepts" " numpy array with two values."
            )
        if np.diff(value) == 0:
            raise ValueError("Values of 'shift_values' must not be the same!")
        self._shift_range = np.asarray(value)

    @property
    def weights(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._weights

    @weights.setter
    def weights(self, value: ty.Optional[ty.Iterable[float]]):
        if value is None:
            value = np.ones(self.n_peaks)
        if not isinstance(value, ty.Iterable):
            raise ValueError("Weights must be provided as an iterable.")
        if len(value) != self.n_peaks:
            raise ValueError("Number of weights does not match the number of peaks.")
        self._weights = np.asarray(value)

    def _initialize(self):
        """Prepare dataset for alignment"""
        # check that values for gaussian_width are valid
        gaussian_widths = np.zeros((self.n_peaks, 1))
        for i in range(self.n_peaks):
            gaussian_widths[i] = self.gaussian_width

        # set the synthetic target signal
        corr_sig_x = np.zeros((self.gaussian_resolution + 1, self.n_peaks))
        corr_sig_y = np.zeros((self.gaussian_resolution + 1, self.n_peaks))

        gaussian_resolution_range = np.arange(0, self.gaussian_resolution + 1)
        for i in range(self.n_peaks):
            left_l = self.peaks[i] - self.gaussian_ratio * gaussian_widths[i]  # noqa
            right_l = self.peaks[i] + self.gaussian_ratio * gaussian_widths[i]  # noqa
            corr_sig_x[:, i] = left_l + (gaussian_resolution_range * (right_l - left_l) / self.gaussian_resolution)
            corr_sig_y[:, i] = self.weights[i] * np.exp(
                -np.square((corr_sig_x[:, i] - self.peaks[i]) / gaussian_widths[i])  # noqa
            )

        self._corr_sig_l = (self.gaussian_resolution + 1) * self.n_peaks
        self._corr_sig_x = corr_sig_x.flatten("F")
        self._corr_sig_y = corr_sig_y.flatten("F")

        # set reduce_range_factor to take 5 points of the previous ranges or half of
        # the previous range if grid_steps < 10
        self._reduce_range_factor = min(0.5, 5 / self.grid_steps)

        # set scl such that the maximum peak can shift no more than the limits imposed by shift when scaling
        self._scale_range = 1 + self.shift_range / max(self.peaks)

        if self._only_shift:
            self._scale_range = np.array([1, 1])

        # create the mesh-grid only once
        mesh_a, mesh_b = np.meshgrid(
            np.divide(np.arange(0, self.grid_steps), self.grid_steps - 1),
            np.divide(np.arange(0, self.grid_steps), self.grid_steps - 1),
        )
        self._search_space = np.tile(
            np.vstack([mesh_a.flatten(order="F"), mesh_b.flatten(order="F")]).T, [1, self._n_iterations]
        )

    def run(self, n_iterations: int = None):
        """Execute the alignment procedure for each signal in the 2D array and collate the shift/scale vectors"""
        self.n_iterations = n_iterations or self.n_iterations
        # iterate for every signal
        t_start = time.time()

        # main loop: searches for the optimum values of Scale and Shift factors by search over a multi-resolution
        # grid, getting better at each iteration. Increasing the number of iterations improves the shift and scale
        # parameters
        for n_signal, y in enumerate(self.array):
            self.shift_opt[n_signal], self.scale_opt[n_signal] = self.compute(y)
        LOGGER.debug(f"Processed {self.n_signals} signals " + time_loop(t_start, self.n_signals + 1, self.n_signals))
        self._computed = True

    def compute(self, y: np.ndarray) -> ty.Tuple[float, float]:
        """Compute correction factors.

        This function does not set value in any of the class attributes so can be used in a iterator where values
        are computed lazily.
        """
        _scale_range = np.array([-0.5, 0.5])
        scale_opt, shift_opt = 0.0, 1.0

        # set to back to the user input arguments (or default)
        _shift = self.shift_range.copy()
        _scale = self._scale_range.copy()

        # generate interpolation function for each signal - instantiation of the interpolator can be quite slow,
        # so you can slightly increase the number of iterations without significant slowdown of the process
        func = generate_function(self.method, self.x, y)

        # iterate to estimate the shift and scale - at each iteration, the grid search is readjusted and the
        # shift/scale values are optimized further
        for n_iter in range(self.n_iterations):
            # scale and shift search space
            scale_grid = _scale[0] + self._search_space[:, (n_iter * 2) - 2] * np.diff(_scale)
            shift_grid = _shift[0] + self._search_space[:, (n_iter * 2) + 1] * np.diff(_shift)
            temp = (
                np.reshape(scale_grid, (scale_grid.shape[0], 1)) * np.reshape(self._corr_sig_x, (1, self._corr_sig_l))
                + np.tile(shift_grid, [self._corr_sig_l, 1]).T
            )
            # interpolate at each iteration. Need to remove NaNs which can be introduced by certain (e.g.
            # PCHIP) interpolator
            temp = np.nan_to_num(func(temp.flatten("C")).reshape(temp.shape))

            # determine the best position
            i_max = np.dot(temp, self._corr_sig_y).argmax()

            # save optimum value
            scale_opt = scale_grid[i_max]
            shift_opt = shift_grid[i_max]

            # readjust grid for next iteration_reduce_range_factor
            _scale = scale_opt + _scale_range * np.diff(_scale) * self._reduce_range_factor
            _shift = shift_opt + _scale_range * np.diff(_shift) * self._reduce_range_factor
        return shift_opt, scale_opt


    def apply(self, return_shifts: bool = None):
        """Align the signals against the computed values"""
        if not self._computed:
            warnings.warn("Aligning data without computing optimal alignment parameters", UserWarning)
        self._return_shifts = return_shifts if return_shifts is not None else self._return_shifts

        if self._only_shift:
            self.shift()
        else:
            self.align()

        # return aligned data and shifts
        if self._return_shifts:
            return self.array_aligned, self.shift_values
        # only return data
        return self.array_aligned

    def align(self, shift_opt=None, scale_opt=None):
        """Realign array based on the optimized shift and scale parameters

        Parameters
        ----------
        shift_opt: Optional[np.ndarray]
            vector containing values by which to shift the array
        scale_opt : Optional[np.ndarray]
            vector containing values by which to rescale the array
        """
        t_start = time.time()
        if shift_opt is None:
            shift_opt = self.shift_opt
        if scale_opt is None:
            scale_opt = self.scale_opt

        # realign based on provided values
        for iteration, y in enumerate(self.array):
            # interpolate back to the original domain
            self.array_aligned[iteration] = self._apply(y, shift_opt[iteration], scale_opt[iteration])
        self.shift_values = self.shift_opt

        LOGGER.debug(f"Re-aligned {self.n_signals} signals " + time_loop(t_start, self.n_signals + 1, self.n_signals))

    def _apply(self, y: np.ndarray, shift_value: float, scale_value: float):
        """Apply alignment correction to array `y`."""
        func = generate_function(self.method, (self.x - shift_value) / scale_value, y)
        return np.nan_to_num(func(self.x))

    def shift(self, shift_opt=None):
        """Quickly shift array based on the optimized shift parameters.

        This method does not interpolate but rather moves the data left and right without applying any scaling.

        Parameters
        ----------
        shift_opt: Optional[np.ndarray]
            vector containing values by which to shift the array
        """
        t_start = time.time()
        if shift_opt is None:
            shift_opt = np.round(self.shift_opt).astype(np.int32)

        # quickly shift based on provided values
        for iteration, y in enumerate(self.array):
            self.array_aligned[iteration] = self._shift(y, shift_opt[iteration])
        self.shift_values = shift_opt

        LOGGER.debug(f"Re-aligned {self.n_signals} signals " + time_loop(t_start, self.n_signals + 1, self.n_signals))

    @staticmethod
    def _shift(y: np.ndarray, shift_value: float):
        """Apply shift correction to array `y`."""
        return shift(y, -int(shift_value))



class Aligner_GPU:
    """Main alignment class"""
     
    def __init__(self, 
                 x: cp.ndarray,
                 peaks: ty.Iterable[float], 
                 array: ty.Optional[cp.ndarray], 
                 method: str = "gpu_linear",
                 n_iterations: int = 5,
                 shift_range: ty.Optional[ty.Tuple[int, int]] = None,
                 return_shifts: bool = False,
                 grid_steps: int = 20,
                 resolution: int = 100,
                 width: float = 10,
                 ratio: float = 2.5,
                 only_shift: bool = False,
                 weights: ty.Optional[ty.List[float]] = None,
                 align_by_index: bool = False):
        
        self.x = cp.asarray(x)
        self.peaks = list(peaks)
        self.n_peaks = len(self.peaks)
        self.n_iterations = n_iterations
        
        self.x = cp.asarray(x)
        if array is not None:
            self.array = check_xy(self.x, cp.asarray(array))
        else:
            self.array = cp.empty((0, len(self.x)))
            
        self.array_aligned = cp.zeros_like(self.array)
            
        if shift_range is None:
            shift_range = cp.asarray([-100, 100])
        self.shift_range = shift_range
        self.n_signals = self.array.shape[0]
        
        self._scale_range = 1 + self.shift_range / max(self.peaks)
        self.method = method
        self.grid_steps = grid_steps
        self._only_shift = only_shift
        # If the number of points is equal to 1, then only shift
        if self.n_peaks == 1:
            only_shift = True
        if only_shift and not align_by_index:
            align_by_index = True
            LOGGER.warning("Only computing shifts - changed `align_by_index` to `True`.")

        
        self._reduce_range_factor = min(0.5, 5 / self.grid_steps)
        
        # accessible attributes
        self.scale_opt = cp.ones((self.n_signals, 1), dtype=cp.float32)
        self.shift_opt = cp.zeros((self.n_signals, 1), dtype=cp.float32)
        self.shift_values = cp.zeros_like(self.shift_opt)
        # return shift vector
        self._return_shifts = return_shifts
        
        
        # create the mesh-grid only once
        mesh_a, mesh_b = cp.meshgrid(
            cp.divide(cp.arange(0, self.grid_steps), self.grid_steps - 1),
            cp.divide(cp.arange(0, self.grid_steps), self.grid_steps - 1),
        )
        self._search_space = cp.tile(
            cp.vstack([mesh_a.flatten(order="F"), mesh_b.flatten(order="F")]).T, [1, self._n_iterations]
        )
        self.gaussian_resolution = resolution
        
        self._corr_sig_l = (self.gaussian_resolution + 1) * self.n_peaks
        
        # set the synthetic target signal
        corr_sig_x = cp.zeros((self.gaussian_resolution + 1, self.n_peaks))
        corr_sig_y = cp.zeros((self.gaussian_resolution + 1, self.n_peaks))
        
        self._corr_sig_x = corr_sig_x.flatten("F")
        self._corr_sig_y = corr_sig_y.flatten("F")
        if weights is None:
            weights = np.ones(self.n_peaks)
        self.weights = weights
        
        # align signals by index rather than peak value
        self._align_by_index = align_by_index
        # align by index - rather than aligning to arbitrary non-integer values in the xvals, you can instead
        # use index of those values
        if self._align_by_index:
            self.peaks = convert_peak_values_to_index_gpu(self.x, self.peaks)
            self.x = cp.arange(self.x.shape[0])
            LOGGER.debug(f"Aligning by index - peak positions: {self.peaks}")
            
        self.gaussian_width = width
        self.gaussian_ratio = ratio
    
 
        self._initialize()
    
        
    @property
    def method(self):
        """Interpolation method."""
        return self._method
    
    @method.setter
    def method(self, value: str):
        if value not in METHODS:
            raise ValueError(f"Method `{value}` not found in the method options: {METHODS}")
        self._method = value
        
    @property
    def grid_steps(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._grid_steps

    @grid_steps.setter
    def grid_steps(self, value: int):
        if value < 1 or not isinstance(value, int):
            raise ValueError("Value of 'iterations' must be above 0 and be an integer!")
        self._grid_steps = value
        
    @property
    def n_iterations(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, value: int):
        if value < 1 or not isinstance(value, int):
            raise ValueError("Value of 'iterations' must be above 0 and be an integer!")
        self._n_iterations = value
    
    
    
    
    def run(self, n_iterations: int = None):
        """Execute the alignment procedure for each signal in the 2D array and collate the shift/scale vectors"""
        self.n_iterations = n_iterations or self.n_iterations
        # iterate for every signal
        t_start = time.time()
    
        # main loop: searches for the optimum values of Scale and Shift factors by search over a multi-resolution
        # grid, getting better at each iteration. Increasing the number of iterations improves the shift and scale
        # parameters
        for n_signal, y in enumerate(self.array):
            self.shift_opt[n_signal], self.scale_opt[n_signal] = self.compute(y)
        LOGGER.debug(f"Processed {self.n_signals} signals " + time_loop(t_start, self.n_signals + 1, self.n_signals))
        self._computed = True
        
        
    def run_batch(self, n_iterations: int = None):
        """Execute the alignment procedure for the entire batch of signals and collate the shift/scale vectors."""
        self.n_iterations = n_iterations or self.n_iterations
        t_start = time.time()

        # Process all signals at once using the modified compute function
        self.shift_opt, self.scale_opt = self.compute_batch(self.array)

        # Here, use appropriate logging to account for CuPy/GPU execution if needed
        print(f"Processed {self.array.shape[0]} signals in {time.time() - t_start} seconds")
        self._computed = True
    
    def compute_batch(self, Y: cp.ndarray) -> ty.Tuple[cp.ndarray, cp.ndarray]:
        """
        Compute correction factors for a batch of signals.
        
        :param Y: 2D CuPy array where each row is a signal.
        :return: Two 1D CuPy arrays containing the shift and scale optimizations for each signal.
        """
        _scale_range = cp.array([-0.5, 0.5])
        
        # Initialize arrays to hold the optimal scale and shift for each signal
        scale_opts = cp.zeros(Y.shape[0], dtype=cp.float32)
        shift_opts = cp.ones(Y.shape[0], dtype=cp.float32)
        
        _shift = self.shift_range.copy()
        _scale = self._scale_range.copy()
        
        # Assume generate_function can handle batch processing
        # This will likely need to be adapted
        funcs = generate_function(self.method, self.x, Y)  # This needs to support 2D arrays
        
        for n_iter in range(self.n_iterations):
            scale_grid = _scale[0] + self._search_space[:, (n_iter * 2) - 2] * cp.diff(_scale)
            shift_grid = _shift[0] + self._search_space[:, (n_iter * 2) + 1] * cp.diff(_shift)
            
            # Here, we assume each signal can be interpolated independently in parallel
            # Adjust temp calculation for batch processing
            temp = (cp.reshape(scale_grid, (-1, 1, 1)) * self._corr_sig_x + shift_grid[:, None, :])
            temp = temp.reshape(temp.shape[0], -1)  # Flatten for interpolation, while keeping batch dimension separate
            
            # Interpolate and reshape back to original grid shape for each signal
            temp = cp.nan_to_num(funcs(temp).reshape(Y.shape[0], len(scale_grid), -1))
            
            # Find the best position for each signal
            i_max = cp.argmax(cp.dot(temp, self._corr_sig_y.T), axis=1)
            
            # Update optimum values
            scale_opts = scale_grid[i_max]
            shift_opts = shift_grid[i_max]
            
            # Readjust grid for next iteration
            _scale = scale_opts[:, None] + _scale_range * cp.diff(_scale) * self._reduce_range_factor
            _shift = shift_opts[:, None] + _scale_range * cp.diff(_shift) * self._reduce_range_factor
        
        return shift_opts, scale_opts
        
    def apply(self, return_shifts: bool = None):
        """Align the signals against the computed values"""
        if not self._computed:
            warnings.warn("Aligning data without computing optimal alignment parameters", UserWarning)
        self._return_shifts = return_shifts if return_shifts is not None else self._return_shifts

        if self._only_shift:
            self.shift()
        else:
            self.align()

        # return aligned data and shifts
        if self._return_shifts:
            return self.array_aligned, self.shift_values
        # only return data
        return self.array_aligned
        
    def align(self, shift_opt=None, scale_opt=None):
        """Realign array based on the optimized shift and scale parameters

        Parameters
        ----------
        shift_opt: Optional[np.ndarray]
            vector containing values by which to shift the array
        scale_opt : Optional[np.ndarray]
            vector containing values by which to rescale the array
        """
        t_start = time.time()
        if shift_opt is None:
            shift_opt = self.shift_opt
        if scale_opt is None:
            scale_opt = self.scale_opt

        # realign based on provided values
        for iteration, y in enumerate(self.array):
            # interpolate back to the original domain
            self.array_aligned[iteration] = self._apply(y, shift_opt[iteration], scale_opt[iteration])
        self.shift_values = self.shift_opt
        
        
    def _apply(self, y: cp.ndarray, shift_value: float, scale_value: float):
        """Apply alignment correction to array `y`."""
        func = generate_function(self.method, (self.x - shift_value) / scale_value, y)
        return cp.nan_to_num(func(self.x))
        
    
    def compute(self, y: cp.ndarray) -> ty.Tuple[float, float]:
        """Compute correction factors.
    
        This function does not set value in any of the class attributes so can be used in a iterator where values
        are computed lazily.
        """
        _scale_range = cp.array([-0.5, 0.5])
        scale_opt, shift_opt = 0.0, 1.0
    
        # set to back to the user input arguments (or default)
        _shift = self.shift_range.copy()
        _scale = self._scale_range.copy()
    
        # generate interpolation function for each signal - instantiation of the interpolator can be quite slow,
        # so you can slightly increase the number of iterations without significant slowdown of the process
        func = generate_function(self.method, self.x, y)
    
        # iterate to estimate the shift and scale - at each iteration, the grid search is readjusted and the
        # shift/scale values are optimized further
        for n_iter in range(self.n_iterations):
            # scale and shift search space
            scale_grid = _scale[0] + self._search_space[:, (n_iter * 2) - 2] * cp.diff(_scale)
            shift_grid = _shift[0] + self._search_space[:, (n_iter * 2) + 1] * cp.diff(_shift)
            temp = (
                cp.reshape(scale_grid, (scale_grid.shape[0], 1)) * cp.reshape(self._corr_sig_x, (1, self._corr_sig_l))
                + cp.tile(shift_grid, [self._corr_sig_l, 1]).T
            )
            # interpolate at each iteration. Need to remove NaNs which can be introduced by certain (e.g.
            # PCHIP) interpolator
            temp = cp.nan_to_num(func(temp.flatten("C")).reshape(temp.shape))
    
            # determine the best position
            i_max = cp.dot(temp, self._corr_sig_y).argmax()
    
            # save optimum value
            scale_opt = scale_grid[i_max]
            shift_opt = shift_grid[i_max]
    
            # readjust grid for next iteration_reduce_range_factor
            _scale = scale_opt + _scale_range * cp.diff(_scale) * self._reduce_range_factor
            _shift = shift_opt + _scale_range * cp.diff(_shift) * self._reduce_range_factor
        return shift_opt, scale_opt
   

    def _initialize(self):
        """Prepare dataset for alignment"""
        # check that values for gaussian_width are valid
        gaussian_widths = cp.zeros((self.n_peaks, 1))
        for i in range(self.n_peaks):
            gaussian_widths[i] = self.gaussian_width

        # set the synthetic target signal
        corr_sig_x = cp.zeros((self.gaussian_resolution + 1, self.n_peaks))
        corr_sig_y = cp.zeros((self.gaussian_resolution + 1, self.n_peaks))

        gaussian_resolution_range = cp.arange(0, self.gaussian_resolution + 1)
        for i in range(self.n_peaks):
            left_l = self.peaks[i] - self.gaussian_ratio * gaussian_widths[i]  # noqa
            right_l = self.peaks[i] + self.gaussian_ratio * gaussian_widths[i]  # noqa
            corr_sig_x[:, i] = left_l + (gaussian_resolution_range * (right_l - left_l) / self.gaussian_resolution)
            corr_sig_y[:, i] = self.weights[i] * cp.exp(
                -cp.square((corr_sig_x[:, i] - self.peaks[i]) / gaussian_widths[i])  # noqa
            )

        self._corr_sig_l = (self.gaussian_resolution + 1) * self.n_peaks
        self._corr_sig_x = corr_sig_x.flatten("F")
        self._corr_sig_y = corr_sig_y.flatten("F")

        # set reduce_range_factor to take 5 points of the previous ranges or half of
        # the previous range if grid_steps < 10
        self._reduce_range_factor = min(0.5, 5 / self.grid_steps)

        # set scl such that the maximum peak can shift no more than the limits imposed by shift when scaling
        self._scale_range = 1 + self.shift_range / max(self.peaks)

        if self._only_shift:
            self._scale_range = cp.array([1, 1])

        # create the mesh-grid only once
        mesh_a, mesh_b = cp.meshgrid(
            cp.divide(cp.arange(0, self.grid_steps), self.grid_steps - 1),
            cp.divide(cp.arange(0, self.grid_steps), self.grid_steps - 1),
        )
        self._search_space = cp.tile(
            cp.vstack([mesh_a.flatten(order="F"), mesh_b.flatten(order="F")]).T, [1, self._n_iterations]
        )
        
    def shift(self, shift_opt=None):
        """Quickly shift array based on the optimized shift parameters.

        This method does not interpolate but rather moves the data left and right without applying any scaling.

        Parameters
        ----------
        shift_opt: Optional[np.ndarray]
            vector containing values by which to shift the array
        """
        t_start = time.time()
        if shift_opt is None:
            shift_opt = cp.round(self.shift_opt).astype(cp.int32)

        # quickly shift based on provided values
        for iteration, y in enumerate(self.array):
            self.array_aligned[iteration] = self._shift(y, shift_opt[iteration])
        self.shift_values = shift_opt

        LOGGER.debug(f"Re-aligned {self.n_signals} signals " + time_loop(t_start, self.n_signals + 1, self.n_signals))

    @staticmethod
    def _shift(y: cp.ndarray, shift_value: float):
        """Apply shift correction to array `y`."""
        return shift(y, -int(shift_value))

  
        
  
     
# alinger = Aligner(x = ms_dict['mz'],
#                   peaks = ms_dict['mz'][p2], 
#                   array= ms_dict["I"],
#                   method="gpu_linear")

# alinger.run()

# test_I = alinger.apply()

top_20_p2 = np.argsort(global_spectral_avg_mean[p2])[::-1][:20]
top_20_indices = p2[top_20_p2]

peaks = ms_dict['mz'][top_20_indices]
weights=np.repeat(20,20)


alinger = Aligner_GPU(x = ms_dict['mz'],
                  peaks = ms_dict['mz'][p2], 
                  array= ms_dict["I"][[1,1000,2000,3000,4000,5000,6000]],
                  method="gpu_linear",
                #  weights=weights,
                  align_by_index=True,
                  only_shift=True,
                  return_shifts=True,
                  n_iterations = 50,
                  width = 20,
                  ratio = 10
                  )

alinger.run()

test_I, shifts = alinger.apply()





# start_time = time.time()
# # instantiate aligner object
# ms_dict["I"] = Aligner(
#     x,
#     ms_dict["I"],
#     peaks,
#     weights=weights,
#     return_shifts=True,
#     align_by_index=True,
#     only_shift=True,
#     method="pchip",
# )

#    # np.zeros((100,100))
   
   
gc.collect()

cp.get_default_memory_pool().free_all_blocks()


import cProfile
cProfile.run("alinger.run()")
cProfile.run("alinger.apply()")


pr = cProfile.Profile()
pr.enable()
alinger.run()
pr.disable()
pr.dump_stats('nan_gpu_slow_down.prof')

p = pstats.Stats('nan_gpu_slow_down.prof')
p.sort_stats('cumulative')
p.print_callers('._check_nan_inf')




#========VISUALIZATION OF ALIGNED RESULTS================#
from msalign.utilities import find_nearest_index
# ========================Lukas's code===================
def overlay_plot(ax, x, array, peak):
    """Generate overlay plot, showing each signal and the alignment peak(s)"""
    for i, y in enumerate(array):
        print(i)
        y = (y / y.max()) + (i * 0.2)
        ax.plot(x, y, lw=3)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_xlabel("Index", fontsize=18)
    ax.set_xlim((x[0], x[-1]))
    ax.vlines(peak, *ax.get_ylim())

def plot_peak(ax, x, y, peak, window=100):
    peak_idx = find_nearest_index(x, peak)
    _x = x[peak_idx-window:peak_idx+window]
    _y = y[peak_idx-window:peak_idx+window]
    ax.plot(_x, _y)

    ax.axes.get_yaxis().set_visible(False)
    #ax.set_xlim((_x[0], _x[-1]))
    ax.set_xlim((peak - 0.1, peak + 0.1))  # Adjusted xlim
    ax.vlines(peak, *ax.get_ylim())

def zoom_plot(axs, x, array, aligned_array, peaks):
    for i, y in enumerate(array):
        for j, peak in enumerate(peaks):
            plot_peak(axs[0, j], x, y, peak)

    for i, y in enumerate(aligned_array):
        for j, peak in enumerate(peaks):
            plot_peak(axs[1, j], x, y, peak)
#=============================================#

# display before and after shifting
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
overlay_plot(ax[0], ms_dict['mz'], ms_dict['I'][:20], ms_dict['mz'][p2])
overlay_plot(ax[1], ms_dict['mz'], test_I.get(), ms_dict['mz'][p2])

# zoom-in on each peak
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 10))
zoom_plot(ax, ms_dict['mz'], ms_dict['I'][[1,1000,2000,3000,4000,5000,6000]], 
          test_I.get(), 
          ms_dict['mz'][p2][100:104])





#=============Performance Benchmarking==============#


import cProfile
import pstats
import io
import pandas as pd
 


interpolator_t = []
searchsort_t = []
compute_t = []
hardware_type = []
 

for hardware in ['gpu']:
    
        
    for spectra_num in range(3000, 5001, 1000):
        hardware_type.append(hardware)
        
        
        if hardware == 'gpu':
            method_time = 'gpu_linear'
            aligner = Aligner_GPU(x = ms_dict['mz'],
                              peaks = ms_dict['mz'][p2], 
                              array= ms_dict["I"][:spectra_num],
                              method=method_time)
        else:
            method_time = 'linear'
            aligner = Aligner_CPU(x = ms_dict['mz'],
                              peaks = ms_dict['mz'][p2], 
                              array= ms_dict["I"][:spectra_num],
                              method=method_time)
        
        
    
        pr = cProfile.Profile()
        pr.enable()
        aligner.run()
        pr.disable()
         
        result = io.StringIO()
        pstats.Stats(pr,stream=result).print_stats()
        result=result.getvalue()
        # chop the string into a csv-like buffer
        result='ncalls'+result.split('ncalls')[-1]
        result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
        # save it to disk
        
        with open('test.csv', 'w+') as f:
            #f=open(result.rsplit('.')[0]+'.csv','w')
            f.write(result)
            f.close()
            
        # Open the CSV file in read mode
        with open('test.csv', 'r') as file:
            for line in file:
            # Check if the search character is in the current line
                if "(interpolator)" in line or "(_call_linear)" in line:
                    # If so, do something with the line, like print it or add it to a list
                    line_char = line.strip()  # Print the line
                    t = float(line_char.split(',')[1])
                    
                    interpolator_t.append(t)  # Add the line to the list
                    
                    
                if "(compute)" in line:
                    # If so, do something with the line, like print it or add it to a list
                    line_char = line.strip()  # Print the line
                    t = float(line_char.split(',')[1])
                    
                    compute_t.append(t)  # Add the line to the list
                    
            
                if "method 'searchsorted' of 'numpy.ndarray' objects" in line or "method 'argmax' of 'cupy._core.core._ndarray_base' objects" in line:
                    # If so, do something with the line, like print it or add it to a list
                    line_char = line.strip()  # Print the line
                    t = float(line_char.split(',')[1])
                    
                    searchsort_t.append(t)  # Add the line to the list
                    
        print(cp.get_default_memory_pool().used_bytes())  # GPU memory
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        print(hardware)
        print(searchsort_t)
        print(compute_t)
        print(interpolator_t)
        





#=============Performance Benchmarking==============#
        
        
fig, ax = plt.subplots(figsize=(10, 7))

# Colors for each category
colors = ['blue', 'red', 'purple']

benchmark_data = np.stack((searchsort_t, compute_t, interpolator_t), axis=0)
max_values = np.max(benchmark_data, axis=1)
normalized_scores = benchmark_data / max_values.reshape((3,-1)) * 100


categories = ['search_sort', 'compute', 'interpolator']
markers = {'cpu': 'o', 'gpu': 's'}

# Create a scatter plot for each category
for i, (category, color) in enumerate(zip(categories, colors)):
    # Assuming the x-values are the indices of each data point
    x_values = np.concatenate((np.arange(10,101,10),np.arange(10,101,10),np.arange(200,1001,100),np.asarray([2000])))
    
    # The y-values are the data points themselves
    y_values = normalized_scores[i, :]
    
        # Plot each point individually to assign marker based on class
    for x, y, label in zip(x_values, y_values, hardware_type):
        ax.scatter(x, y, color=color, marker=markers[label], label=f'{label}' if x == 0 else "")
        
    ax.plot(x_values[np.asarray(hardware_type)=="gpu"], y_values[np.asarray(hardware_type)=="gpu"], color=color)
    ax.plot(x_values[np.asarray(hardware_type)!="gpu"], y_values[np.asarray(hardware_type)!="gpu"], color=color)


# Adding labels and title
plt.xlabel('Number of Spectra')
plt.ylabel('Max-Normalized Time [%]')
#plt.title('Scatter Plot of 4 Categories')

import matplotlib.lines as mlines
# Create custom legend entries for CPU and GPU
legend_entries = []
for category, color in zip(categories, colors):
    legend_entries.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=category))
legend_entries.append(mlines.Line2D([], [], color='none', marker='o', markeredgecolor='black', linestyle='None', markersize=10, label='CPU'))
legend_entries.append(mlines.Line2D([], [], color='none', marker='s', markeredgecolor='black', linestyle='None', markersize=10, label='GPU'))

# Apply legend
ax.legend(handles=legend_entries, fontsize="large", loc='upper right')

ax.tick_params(axis='x', rotation=0)
plt.xlabel('Number of Spectra')
plt.ylabel('Max-Normalized Time [%]')
#plt.show()

plt.savefig("Z:\\Lin\\2023\\Python\\Scripts\\msalign\\figures\\mass_align_CPU_v_GPU", dpi=200)     

