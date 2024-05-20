# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:47:40 2024

@author: 6680e310
"""

#========= LOAD SOME NICE LIBRARIES ==========#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
mpl.rcParams['font.size'] = 20
import cv2
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from numba import njit, jit
import psutil
import scipy
from scipy.signal import find_peaks, peak_widths
from bokeh.plotting import figure, show, save, output_file

#from align_gpu
import warnings
import logging
import typing as ty
import time
import os
import time
import gc


try:
    import cupy as cp
    print(f"cupy is installed and imported successfully!")
except ImportError:
    print(f"cupy is not installed or could not be imported.")

os.chdir("/home/yutinlin/z_drive/Lin/2023/Python/Scripts/msalign")
from utilities import check_xy, convert_peak_values_to_index, convert_peak_values_to_index_gpu, generate_function, shift, time_loop, find_nearest_index, find_nearest_index_gpu
import ImzMLParser_chunk
from ImzMLParser_chunk import ImzMLParser_chunk

from nan_to_num_njit_cupy import fillna_cpwhere_njit


import ray
if not ray.is_initialized():
    ray.init(num_cpus=10)

    

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
    
    
def significance(pvalue):
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

#========================PEAK INTEGRATION========================#

def peak_integration(column, intensity_df, deviation_df, MZ, p2_new, peak_area_df, chunk_0_mz):
    k=0
    for value in p2_new:
        print(value)
        count_left = 0
        count_right = 0
        if value:
            value_left = value
            value_right = value
            # while df_test.loc[value_left, 'deviation']=="False":  #old code where count = 3 is True was previously defined
            while count_left < count_rep:
                value_left -= 1
                if deviation_df[column][value_left] == True:
                    count_left += 1
                else:
                    count_left = 0
                if count_left == count_rep:
                    value_left += (count_rep-1+2)# +2 to rely more on symmetry for better peaks at deviating regions
            # while df_test.loc[value_right, 'deviation']=="False":
            while count_right < count_rep:
                value_right += 1
                if deviation_df[column][value_right] == True:
                    count_right += 1
                else:
                    count_right = 0
                if count_right == count_rep:
                    value_right -= (count_rep-1+2)
         

            #value_left += 2
            #value_right -= 2
            extra_bins = abs((value-value_left)-(value_right-value))
            # +1 adjustment to account for delays in derivatives
            if extra_bins > 0:
                peak_area = (np.trapz(intensity_df[column][value_left:value], MZ[value_left:value]) +
                             np.trapz(intensity_df[column][value:value_right-1], MZ[value:value_right-1]) +
                             np.trapz(intensity_df[column][value_left:(value_left+extra_bins+1)], MZ[value_left:(value_left+extra_bins+1)]))
                # peak_area_df.at[MZ[value], 'left_width'] += MZ[value_left]
                # peak_area_df.at[MZ[value], 'right_width'] += MZ[value_right-1]
            elif extra_bins < 0:
                peak_area = (np.trapz(intensity_df[column][value_left+1:value], MZ[value_left+1:value]) +
                             np.trapz(intensity_df[column][value:value_right], MZ[value:value_right]) +
                             np.trapz(intensity_df[column][(value_right+extra_bins-1):value_right], MZ[(value_right+extra_bins-1):value_right]))
                # peak_area_df.at[MZ[value], 'left_width'] += MZ[value_left+1]
                # peak_area_df.at[MZ[value], 'right_width'] += MZ[value_right]
            else:
                peak_area = (np.trapz(intensity_df[column][value_left:value], MZ[value_left:value]) +
                             np.trapz(intensity_df[column][value:value_right], MZ[value:value_right]))
                # peak_area_df.at[MZ[value], 'left_width'] += MZ[value_left]
                # peak_area_df.at[MZ[value], 'right_width'] += MZ[value_right]
            peak_area_df.iloc[k, column] = peak_area
            k += 1
  

#========================PROCESSING CHUNKED DATA========================#

def get_chunk_ms_info(p, chunk_start, chunk_size):
    """Create the dictionary obtaining intensity and M/Z values from a chunk of data.

    Parameters
    ----------
    p: pyimzml.ImzMLParser
        The data to process, converted to ImzMLParser from the .imzML raw date file.
    
    chunk_start: int
        The row index location of where the first chunck starts in the  entire data set.

    chunk_size: int
        The size of the chunk to obtain information from.

    Returns
    -------
    ms_dict: dictionary of 2 key:value pairs
        The first key "I" contains value pair {ndarray} of shape {}
        The remaining columns of data after assigning each chunk a minimal number of columns.

    """
    ms_dict = dict()
    ms_dict["I"] = []
    ms_dict["coords"] = []
    index_stop = int(p.getspectrum(0)[0].shape[0]*0.999)
    
    for idx, (x, y, z) in enumerate(p.coordinates[chunk_start:chunk_start+chunk_size]):
        index = chunk_start + idx
        print(index)
        mzs, intensities = p.getspectrum(index)
        ms_dict["mz"] = mzs[0:index_stop]
        ms_dict["I"].append(intensities[0:index_stop])
        ms_dict["coords"].append(p.coordinates[index])
    ms_dict["I"] = np.array(ms_dict["I"])
    ms_dict["coords"] = np.array(ms_dict["coords"])

    return ms_dict


def get_chunk_ms_info2(p, chunk_start, chunk_size):
    """Create the dictionary obtaining intensity and M/Z values from a chunk of data.

    Parameters
    ----------
    p: pyimzml.ImzMLParser
        The data to process, converted to ImzMLParser from the .imzML raw date file.
    
    chunk_start: int
        The row index location of where the first chunck starts in the  entire data set.

    chunk_size: int
        The size of the chunk to obtain information from.

    Returns
    -------
    ms_dict: dictionary of 2 key:value pairs
        The first key "I" contains value pair {ndarray} of shape {}
        The remaining columns of data after assigning each chunk a minimal number of columns.

    """
    ms_dict = dict()
    index_stop = int(p.getmz(0).shape[0]*0.999)
    
    ms_dict["mz"] = p.getmz(0)[:index_stop] #should pixel 0 be chosen? looks like mz is same across indices
    
    ms_dict["I"] = p.get_intensity_chunk(chunk_start=chunk_start, chunk_end=(chunk_start+chunk_size), index_stop=index_stop)
    ms_dict["I"] = ms_dict["I"][:, :index_stop]
    
    ms_dict["coords"] = np.asarray(p.coordinates[chunk_start:(chunk_start+chunk_size)])

    return ms_dict


def get_chunk_ms_info_inhomogeneous(p, chunk_start, chunk_size, max_mz, min_mz, mz_RP, RP, dist, rp_factor):
    """Create the dictionary obtaining intensity and M/Z values from a chunk of data.

    Parameters
    ----------
    p: pyimzml.ImzMLParser
        The data to process, converted to ImzMLParser from the .imzML raw date file.
    
    chunk_start: int
        The row index location of where the first chunck starts in the  entire data set.

    chunk_size: int
        The size of the chunk to obtain information from.

    Returns
    -------
    ms_dict: dictionary of 2 key:value pairs
        The first key "I" contains value pair {ndarray} of shape {}
        The remaining columns of data after assigning each chunk a minimal number of columns.

    """
    ms_dict = dict()
    ms_dict["I"] = []
    ms_dict["coords"] = []
    
    
    for idx, (x, y, z) in enumerate(p.coordinates[chunk_start:chunk_start+chunk_size]):
        bin_spectrum = np.zeros([int((max_mz-min_mz)/(mz_RP/RP) * rp_factor)+10])
        index = chunk_start + idx
        mzs, intensities = p.getspectrum(index)
        intensity_index = np.round((mzs - min_mz) / (mz_RP/RP) * rp_factor, 0).astype(np.int32)
        bin_spectrum[intensity_index] += intensities
        ms_dict["I"].append(bin_spectrum)
        ms_dict["coords"].append(p.coordinates[index])
    ms_dict["I"] = np.array(ms_dict["I"])
    ms_dict["coords"] = np.array(ms_dict["coords"])

    return ms_dict



#========================PREPARE AND CALCULATE P2========================#

def chunk_prep(p, allocate_percent):
    """Return the baseline numbers for chunking which is dependent on the dataset.

    Parameters
    ----------
    p :  pyimzml.ImzMLParser
        The data to process, converted to ImzMLParser from the .imzML raw date file.

    Returns
    -------
    num_chunks: int
        Number of chunks the data needs to be devided into to optimize the use of 
        avilable RAM on the computer.
    
    chunk_size_base: int
        The minimal number of columns in each chunk.

    chunk_start: int
        The row index location of where the first chunck starts in the .

    remainder: int
        The remaining columns of data after assigning each chunk a minimal number of columns.

    """
    row_spectra = len(p.intensityLengths)
    col_spectra = p.intensityLengths[0]

    RAM_available = psutil.virtual_memory()[1]
    num_chunks = int(row_spectra * col_spectra *
                     np.dtype(np.float32).itemsize / (RAM_available * allocate_percent/100)) + 1  # ceiling

    chunk_size_base = row_spectra // num_chunks
 
    remainder = row_spectra % num_chunks

    chunk_start = 0

    return num_chunks, chunk_size_base, chunk_start, remainder


def chunk_prep_inhomogeneous(p, allocate_percent, max_mz, min_mz, mz_RP, RP, dist):
    row_spectra = len(p.coordinates)
    col_spectra = int((max_mz-min_mz)/(mz_RP/RP)*dist)+10

    RAM_available = psutil.virtual_memory()[1]
    num_chunks = int(row_spectra * col_spectra *
                     np.dtype(np.float32).itemsize / (RAM_available * allocate_percent/100)) + 1  # ceiling

    chunk_size_base = row_spectra // num_chunks
 
    remainder = row_spectra % num_chunks

    chunk_start = 0

    return num_chunks, chunk_size_base, chunk_start, remainder





def integrate_peak(chunk_ms_dict, p2_new_temp, p2_width, chunk_0_mz_peaks, method, remove_duplicates=False):
    
    # ===================== DIFFERENT INTEGRATION ALGORITHMS==========================#
    if method == "peak_width":
        peak_width = p2_width[2:].T
        peak_width = np.round(peak_width)
        #peak_width = np.asarray([np.ceil(peak_width[:,0]), np.floor(peak_width[:,1])]).T

        if remove_duplicates:
            duplicates = (peak_width[:-1, -1] == peak_width[1:, 0])
            keep_rows = np.insert(~duplicates, 0, True)
            peak_width = peak_width[keep_rows]
            p2_new_temp = p2_new_temp[keep_rows]
        
        intensity_df_temp = pd.DataFrame(
            chunk_ms_dict["I"][:, 1:].T, index=chunk_ms_dict['mz'][1:].T)
        intensity_df_temp.index.name = 'mz'
        intensity_df_temp.reset_index(inplace=True)
        
        peak_area_df_temp = pd.DataFrame(np.zeros(
            (len(p2_new_temp), intensity_df_temp.shape[1]-1)), index=chunk_0_mz_peaks)  # df[1:] to account for np.diff

        for column in intensity_df_temp.columns[1:]:
            peak_area = [np.trapz(chunk_ms_dict["I"][column][np.arange(int(row[0]), int(row[1]) + 1)], chunk_ms_dict["mz"][np.arange(int(row[0]), int(row[1]) + 1)]-chunk_ms_dict["mz"][np.arange(int(row[0])-1, int(row[1]))]) for row in peak_width]
            peak_area_df_temp.iloc[:,column] = np.asarray(peak_area)

    #================================================#
    elif method == "derivatives":
        dtype = np.float32
        firstderivative_array = np.zeros(
            chunk_ms_dict["I"][:, 1:].shape, dtype=dtype)
        secondderivative_array = np.zeros(
            chunk_ms_dict["I"][:, 1:].shape, dtype=dtype)
    
        # process data in chunk
        iter_I = iter(chunk_ms_dict["I"])
    
        try:
            row_index = 0
            while True:
                row = next(iter_I)  # Read the next row (AKA pixel)
                row_firstderivative = np.gradient(
                    row, chunk_ms_dict['mz']).astype(np.float32)
                row_firstderivative_smooth = scipy.signal.savgol_filter(
                    row_firstderivative, window_length=10, polyorder=3).astype(np.float32)
                row_secondderivative = np.gradient(
                    row_firstderivative_smooth, chunk_ms_dict['mz']).astype(np.float32)
                row_secondderivative_smooth = scipy.signal.savgol_filter(
                    row_secondderivative, window_length=20, polyorder=4).astype(np.float32)
                # Append the processed chunk to the memmap array
                firstderivative_array[row_index, :] = np.diff(
                    row_firstderivative_smooth)
                secondderivative_array[row_index, :] = np.diff(
                    row_secondderivative_smooth)
                row_index += 1
        except StopIteration:
            print("Reached the end of the chunk.")
    
        #================================================#

        # start from one bc of np.diff()
        intensity_df_temp = pd.DataFrame(
            chunk_ms_dict["I"][:, 1:].T, index=chunk_ms_dict['mz'][1:].T)
        intensity_df_temp.index.name = 'mz'
    
        gc.collect()
    
        deviation_df_temp = ((secondderivative_array > 0) & (firstderivative_array > 0) & (np.diff(chunk_ms_dict['I'], axis=1) > 0) |
                             (secondderivative_array < 0) & (firstderivative_array > 0) & (np.diff(chunk_ms_dict['I'], axis=1) < 0) |
                             (chunk_ms_dict['I'][:, 1:] <= 3*noise))
    
        gc.collect()
        left_width = []
        right_width = []
        peak_area_array = []
        count_rep = 3
        intensity_df_temp.reset_index(inplace=True)
    
        peak_area_df_temp = pd.DataFrame(np.zeros(
            (len(p2_new_temp), intensity_df_temp.shape[1]-1)), index=chunk_0_mz_peaks)  # df[1:] to account for np.diff

    
        for column in intensity_df_temp.columns[1:]:
            peak_integration(column, intensity_df_temp, deviation_df_temp,
                             chunk_0_mz, p2_new_temp, peak_area_df_temp, chunk_0_mz_peaks)  # axis 1 to colunmn operate
            # peak_area_df is updated in the function
        del firstderivative_array, secondderivative_array
        # ===================== DIFFERENT INTEGRATION ALGORITHMS==========================#
    else: 
        print("method DNE")
        
    return peak_area_df_temp

def get_p2(p, loq, lwr, upr, rel_height, chunk_size_base, num_chunks, chunk_start, remainder, dist, noise=0, method="point"):
    index_stop = int(p.getmz(0).shape[0]*0.999)
    avg_spectrum = np.zeros(index_stop)
    mzs = p.getmz(0)[:index_stop] #should pixel 0 be chosen? looks like mz is same across indices
    
    for i in range(num_chunks):
        print(f'Chunk {i}')
        start_time = time.time()
        chunk_size_temp = chunk_size_base
        if remainder > i:
            chunk_size_temp += 1
        
        intensities_chunk = p.get_intensity_chunk(chunk_start=chunk_start, chunk_end=(chunk_start+chunk_size_temp), index_stop=index_stop)
        avg_spectrum += np.sum(intensities_chunk, axis=0)

        if method == "point":
            noise_bin = np.std(np.asarray(intensities_chunk)[:,np.logical_and(mzs >= lwr, mzs <= upr)], axis=0)
            if np.sum(noise_bin) == 0:
                pass
            else:
                noise += np.mean(noise_bin[noise_bin!=0]) 
        
            chunk_start += chunk_size_temp

        print(f"Time used for running chunk {i} with a size of {chunk_size_temp}: {time.time()-start_time}")
        gc.collect()

    avg_spectrum /= len(p.coordinates)
    
    if method == "point":
        noise /= num_chunks
    elif method == "specify_noise":
        noise = noise

    p2, info = find_peaks(x=avg_spectrum, height=loq*noise, distance=dist)
    p2_width = peak_widths(avg_spectrum, p2, rel_height=rel_height)
    p2_width = np.asarray(p2_width)

    return p2, p2_width, noise, mzs, avg_spectrum



#===from transfer_learning_python===#

def remapping_coords(x ,y):
    '''
    Map imzML x, y coordinates to x-min, y-min = 0 with stepwise increase of 1
    -----
    inputs:
        x: x-coordinates from imzML preprocessed matrix
        y: y-coordinates from imzML preprocessed matrix
    -----
    outputs:
        remapped x, y coordinates
    '''
    
#    x = x.to_numpy()
#    y = y.to_numpy()
    a,b = 0,0
    for i in np.sort(np.unique(x)):
        x[x == i] = a
        a += 1
    for j in np.sort(np.unique(y))[::-1]:
        y[y == j] = b
        b += 1
        
    return x,y        


def make_image(matrix, x ,y):
    '''
    Construct images with RGB channels and single channel. RGB conversion to rainbow scheme from grayscale image.
    -----
    inputs:
        matrix: intensity matrix of molecular images as vectors in axis 0
        x: remapped x-coordinates
        y: remapped y-coordinates
    -----
    outputs:
        RGB-transformed and grayscale images
    '''
    
    x_max = max(x)
    y_max = max(y)
    x_min = min(x)
    y_min = min(y)
    
    img_array_3c = np.empty([matrix.shape[1], x_max+1-x_min, y_max+1-y_min, 3])
    img_array_1c = np.empty([matrix.shape[1], x_max+1-x_min, y_max+1-y_min, 1])
    print(matrix.shape[0])
    
    for k in range(matrix.shape[1]-2):
        print(k)

        try:
            color_img = cv2.applyColorMap(matrix[matrix.columns[k]].to_numpy().astype(np.uint8), cv2.COLORMAP_RAINBOW)
        except:
            color_img = np.stack([matrix[matrix.columns[k]].to_numpy().astype(np.uint8)] * 3, axis=-1)
        
        for row, coord in matrix[matrix.columns[[-2, -1]]].iterrows():
            row = int(row)
            img_array_1c[k, coord['x']-x_min, coord['y']-y_min, 0:1] = matrix[matrix.columns[k]][row] 
            img_array_3c[k, coord['x']-x_min, coord['y']-y_min, 0:3] = color_img[row, 0] #0:2 RGB
        
    if np.isnan(img_array_1c).any():
        print('Nan exists in single-channel images')
    elif np.isnan(img_array_3c).any():
        print('Nan exists in rainbow images')

    return img_array_3c, img_array_1c


def make_image_1c(data_2darray, remap_coord = True, max_normalize=True):
    '''
    Construct images with RGB channels and single channel. RGB conversion to rainbow scheme from grayscale image.
    -----
    inputs:
        matrix: intensity matrix of molecular images as vectors in axis 0
        x: remapped x-coordinates
        y: remapped y-coordinates
    -----
    outputs:
        RGB-transformed and grayscale images
    '''
    
    x_min = min(data_2darray[:,-2])
    y_min = min(data_2darray[:,-1])
    
    if remap_coord:
        data_2darray[:,-2] = data_2darray[:,-2] - x_min
        data_2darray[:,-1] = data_2darray[:,-1] - y_min 
    
    img_array_1c = np.zeros([data_2darray.shape[1]-2, int(max(data_2darray[:,-1]))+1, 
                                                    int(max(data_2darray[:,-2]))+1, 1])
    
    if max_normalize:
        data_2darray[:,:-2] = data_2darray[:,:-2] / np.max(data_2darray[:,:-2],axis=0)
    
    
    for k in range(data_2darray.shape[1]-2):

        for row, coord in enumerate(data_2darray[:,-2:]):
            row = int(row)
            img_array_1c[k, int(coord[1]), int(coord[0]), 0:1] = data_2darray[row,k]

    if np.isnan(img_array_1c).any():
        print('Nan exists in single-channel images')


    return img_array_1c


def max_normalize(data_2darray):
    data_2darray = data_2darray / np.max(data_2darray,axis=0)
    return data_2darray


@jit(nopython=True)
def make_image_1c_njit(data_2darray, img_array_1c, remap_coord = True):
    '''
    
    -----
    inputs:
        matrix: intensity matrix of molecular images as vectors in axis 0
        img_array_1c: run img_array_1c = np.empty([data_array.shape[1], int(max(data_array[:,-2])), 
                                                  int(max(data_array[:,-1])), 1])
        y: remapped y-coordinates
    -----
    outputs:
        RGB-transformed and grayscale images
    '''
        
    x_min = min(data_2darray[:,-2])
    y_min = min(data_2darray[:,-1])
    
    if remap_coord:
        data_2darray[:,-2] = data_2darray[:,-2] - x_min
        data_2darray[:,-1] = data_2darray[:,-1] - y_min 
    

    for k in range(data_2darray.shape[1]-2):

        for row, coord in enumerate(data_2darray[:,-2:]):
            row = int(row)
            img_array_1c[k, int(coord[1]), int(coord[0]), 0:1] = data_2darray[row,k]

    if np.isnan(img_array_1c).any():
        pass

    return img_array_1c


def draw_ROI(ROI_info, show_ROI=True, show_square=False, linewidth=3):
    start_col = np.where(ROI_info.columns == "bottom")[0][0]
    truncate_col = np.where(ROI_info.columns == "right")[0][0]
    squares = ROI_info.iloc[:,start_col:truncate_col+1].to_numpy() 
 
    for i, square in enumerate(squares):
        bottom, top, left, right = square
        # plt.ylim(top-10, bottom+10)
        # plt.xlim(left-10, right+10)
        
        x_coords = [left, left, right, right, left]
        y_coords = [bottom, top, top, bottom, bottom]
        if show_square:
            plt.plot(x_coords, y_coords, 'g-', linewidth=linewidth)  
        if show_ROI:
            plt.text((left+right)/2, bottom-3, ROI_info["ROI"][i], 
                     horizontalalignment='center', size=max(ROI_info['right'])/8, color='white')
            

def build_effnet(image_array):
    '''
    Build a CNN feature extractor using EfficientNetV2L
    -----
    inputs:
        image_array: RGB images from make_image()
    -----
    outputs:
        EfficientNetV2L feature extractor
    '''
    
    model = Sequential()

    model.add(Input(shape=(image_array.shape[1:4])))

    model.add(layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))) #resizes the images to the expected size


    en = keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False, weights="imagenet") 
    en.trainable = False # DISABLE training of this entire layer 

    model.add(en)

    model.add(layers.GlobalAveragePooling2D()) # !

    return model


def gromov_trick(image_array):
    '''
    Preprocesses each vector of images in original or CNN space for Euclidean metric
    -----
    inputs:
        image_array: extracted features from transfer learning or img_array_1c from make_image()
    -----
    outputs:
        normalized image_array for k-means clustering
    '''

    image_array_pos = image_array-np.min(image_array, axis=1).reshape((-1,1))
    normalized_image_array = np.sqrt(np.divide(image_array_pos, np.sum(image_array_pos,axis=1, keepdims=True)))
    
    return normalized_image_array



def clustering_in_embedding(image_array, k, compute_index = False):
    '''
    Performs k-means clustering and t-SNE dimensionality reduction. Currently fixed parameters.
    -----
    inputs:
        image_array: normalized images from gromov_trick() or extracted features from transfer learning or img_array_1c from make_image()
    -----
    outputs:
        k-means labels and t-SNE embedding of image_array
    '''
    np.random.seed(0)
    #k-means
    k_means = KMeans(n_clusters=k, init = "k-means++", random_state=0, n_init="auto").fit(image_array)
    kmeans_labels = k_means.labels_
    
    if compute_index:
        DB_score = davies_bouldin_score(image_array, kmeans_labels)
        CH_score = calinski_harabasz_score(image_array, k_means.labels_)
        sil_score = silhouette_score(image_array, k_means.fit_predict(image_array))
        elbow_score = k_means.inertia_
        
        score_dict = {"DB_score": DB_score, "CH_score": CH_score, "sil_score": sil_score, "elbow_score": elbow_score}
    else:
        score_dict = []
    
    #tSNE
    num_iters = 500
    perplexity = 5
    my_random_state = 1
    tsne = TSNE(n_components = 2, n_iter = num_iters,
                  perplexity = perplexity, random_state = my_random_state) # how does the perplexity param change the results?
    tsne_embedding = tsne.fit_transform(image_array)
    
    if compute_index:
        return kmeans_labels, tsne_embedding, score_dict
    else: 
        return kmeans_labels, tsne_embedding, score_dict



def plot_image_at_point(im, xy, zoom, color_scheme="inferno"):
  '''
  Plots a tiny image at point xy for visualization with dimensionally reduced embedding.
  '''
  
  # dxy = np.random.rand(int(np.floor(dxy)))/50 * plt.ylim()
  # plt.arrow(*xy, *dxy)
  ab = AnnotationBbox(OffsetImage(im, zoom=zoom, cmap = color_scheme), xy, frameon=False)
  plt.gca().add_artist(ab)
  
  
  

def get_spectrum(output_filepath, mzs, avg_intensity, p2):
    output_file(filename=output_filepath)
    
    ms_spectrum = figure(width=1400, height=600, title="Interactive mass spectrum")
    ms_spectrum.line(mzs, avg_intensity)
    ms_spectrum.circle(mzs[p2], avg_intensity[p2], color="red", alpha=0.6)
    ms_spectrum.xaxis.axis_label = "m/z"
    ms_spectrum.yaxis.axis_label = "Intensity [a.u.]"
    ms_spectrum.xaxis.axis_label_text_font_size = "26pt"  
    ms_spectrum.yaxis.axis_label_text_font_size = "26pt"  
    # Customize ticks
    ms_spectrum.xaxis.major_tick_line_color = "black"
    ms_spectrum.xaxis.major_tick_line_width = 3
    ms_spectrum.xaxis.major_tick_in = 12  
    ms_spectrum.xaxis.major_tick_out = 6  
    ms_spectrum.yaxis.major_tick_line_color = "black"
    ms_spectrum.yaxis.major_tick_line_width = 3
    ms_spectrum.yaxis.major_tick_in = 12
    ms_spectrum.yaxis.major_tick_out = 6
    ms_spectrum.xaxis.minor_tick_line_color = "grey"
    ms_spectrum.yaxis.minor_tick_line_color = "grey"
    ms_spectrum.xaxis.minor_tick_line_width = 1
    ms_spectrum.yaxis.minor_tick_line_width = 1
    ms_spectrum.xaxis.minor_tick_in = 6
    ms_spectrum.xaxis.minor_tick_out = 3
    ms_spectrum.yaxis.minor_tick_in = 6
    ms_spectrum.yaxis.minor_tick_out = 3
    ms_spectrum.xaxis.major_label_text_font_size = "16pt"
    ms_spectrum.yaxis.major_label_text_font_size = "16pt"
    
    save(ms_spectrum)
  
  

  
import matplotlib.pyplot as plt
import IPython.display as Disp
from ipywidgets import widgets
import numpy as np
import cv2
  
class bbox_select():
    def __init__(self,im):
        self.move = False
        self.im = im
        self.selected_points = []
        self.img = plt.imshow(self.im)
        self.ka = self.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show(block=False)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        Disp.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)

        
    def poly_img(self,img,pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)),7)
        return img

    def onclick(self, event):
    #display(str(event))
        self.selected_points.append([event.xdata,event.ydata])
        if len(self.selected_points)>1:
            self.fig
            self.img.set_data(self.poly_img(self.im.copy(),self.selected_points))
            self.fig.canvas.draw()
            self.move = True
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)
