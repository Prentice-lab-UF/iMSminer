# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 01:06:35 2024

@author: yutin
"""




grid_steps = 20
_n_iterations = 5
resolution = 100
n_peaks = len(ms_dict['mz'][p2])



mesh_a, mesh_b = cp.meshgrid(
    cp.divide(cp.arange(0, grid_steps), grid_steps - 1),
    cp.divide(cp.arange(0, grid_steps), grid_steps - 1),
)
_search_space = cp.tile(
    cp.vstack([mesh_a.flatten(order="F"), mesh_b.flatten(order="F")]).T, [1, _n_iterations]
)
gaussian_resolution = resolution

_corr_sig_l = (gaussian_resolution + 1) * n_peaks

# set the synthetic target signal
corr_sig_x = cp.zeros((gaussian_resolution + 1, n_peaks))
corr_sig_y = cp.zeros((gaussian_resolution + 1, n_peaks))

_corr_sig_x = corr_sig_x.flatten("F")



scale_grid = _scale[0] + _search_space[:, (n_iter * 2) - 2] * cp.diff(_scale)
shift_grid = _shift[0] + _search_space[:, (n_iter * 2) + 1] * cp.diff(_shift)

temp = (
    cp.reshape(scale_grid, (scale_grid.shape[0], 1)) * cp.reshape(_corr_sig_x, (1, _corr_sig_l))
    + cp.tile(shift_grid, [_corr_sig_l, 1]).T
)


func = generate_function("gpu_linear", ms_dict['mz'], ms_dict["I"][1]])
temp = fillna_cpwhere_njit(func(temp.flatten("C")).reshape(temp.shape))
    