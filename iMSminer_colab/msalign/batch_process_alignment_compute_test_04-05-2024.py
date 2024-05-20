# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:35:03 2024

@author: yutin
"""

def generate_function(method, x, y):
    """
    Generate interpolation function

    Parameters
    ----------
    method : str
        name of the interpolator
    x : np.array
        1D array of separation units (N)
    y : np.ndarray
        1D array of intensity values (N)

    Returns
    -------
    fcn : scipy interpolator
        interpolation function
    """
    if method == "pchip":
        return interpolate.PchipInterpolator(x, y, extrapolate=False)
    if method == "gpu_linear":
        return create_interpolator(x, y)
    return interpolate.interp1d(x, y, method, bounds_error=False, fill_value=0)


def create_interpolator(xp, fp, left=None, right=None, period=None):
    """
    Creates an interpolator function for given control points and their values.
    
    Parameters:
    xp (cupy.ndarray): Known x-coordinates.
    fp (cupy.ndarray): Function values at known points xp.
    left (float or complex): Value to return for x < xp[0]. Defaults to fp[0].
    right (float or complex): Value to return for x > xp[-1]. Defaults to fp[-1].
    period (float): Period for x-coordinates. If specified, left and right are ignored.
    
    Returns:
    A function that accepts a cupy.ndarray x and returns interpolated values.
    """
    def interpolator(x):
        return cp.interp(x, xp, fp, left=None, right=None, period=None)
    return interpolator



test_func = generate_function(method="gpu_linear", x= ms_dict["mz"], y= ms_dict["I"][1])

test_func()


shift_range = [-100, 100]

n_signals = ms_dict["I"].shape[0]

_scale_range = 1 + shift_range / max(ms_dict["mz"][p2])

_shift = shift_range.copy()
_scale = _scale_range.copy()


grid_steps=20
n_iter =0
n_iterations = 5
mesh_a, mesh_b = np.meshgrid(
    np.divide(np.arange(0, grid_steps), grid_steps - 1),
    np.divide(np.arange(0, grid_steps), grid_steps - 1),
)
_search_space = cp.tile(
    cp.vstack([mesh_a.flatten(order="F"), mesh_b.flatten(order="F")]).T, [1, n_iterations ]
)

scale_grid = _scale[0] + _search_space[:, (n_iter * 2) - 2] * cp.diff(_scale)
shift_grid = _shift[0] + _search_space[:, (n_iter * 2) + 1] * cp.diff(_shift)

# Here, we assume each signal can be interpolated independently in parallel
# Adjust temp calculation for batch processing
n_peaks = len(ms_dict["mz"][p2])
gaussian_resolution = 100 
_corr_sig_x = cp.zeros((gaussian_resolution + 1, n_peaks))
_corr_sig_y = cp.zeros((gaussian_resolution + 1, n_peaks))
temp = (cp.reshape(scale_grid, (-1, 1, 1)) * _corr_sig_x + shift_grid[:, None, :])
temp = temp.reshape(temp.shape[0], -1)  # Flatten for interpolation, while keeping batch dimension separate




