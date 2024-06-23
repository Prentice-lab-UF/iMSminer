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
import logging
import time
import typing as ty
import warnings
from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import scipy
import scipy.interpolate as interpolate
from bokeh.plotting import figure, output_file, save, show
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from numba import jit
from scipy.signal import find_peaks, peak_widths

try:
    import cupy as cp
except ModuleNotFoundError:
    pass


def significance(pvalue):
    """Conversion to symbolic representation of p-value thresholds"""
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def get_chunk_ms_info(p, chunk_start, chunk_size):
    """Creates a dictionary of intensity, m/z, and coordinate values by iterating through a chunk of imzML data

    Parameters
    ----------
    p : pyimzml.ImzMLParser
        Instace of ImzMLParser class containing imported imzML data

    chunk_start : int
        Pixel index for the start of chunk

    chunk_size : int
        Number of pixels contained in each chunk

    Returns
    -------
    ms_dict : dict
        Key `I` contains the intensity matrix of axes pixel index and m/z bin index
        Key `mz` contains m/z values
        Key `coords` contains pixel indices and coordinate values
    """
    ms_dict = dict()
    ms_dict["I"] = []
    ms_dict["coords"] = []
    index_stop = int(p.getspectrum(0)[0].shape[0] * 0.999)

    for idx, (x, y, z) in enumerate(
        p.coordinates[chunk_start: chunk_start + chunk_size]
    ):
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
    """Creates a dictionary of intensity, m/z, and coordinate values from a chunk of imzML data

    Parameters
    ----------
    p : pyimzml.ImzMLParser
        Instace of ImzMLParser class containing imported imzML data

    chunk_start : int
        Pixel index for the start of chunk

    chunk_size : int
        Number of pixels contained in each chunk

    Returns
    -------
    ms_dict : dict
        Key `I` contains the intensity matrix of axes pixel index and m/z bin index
        Key `mz` contains m/z values
        Key `coords` contains pixel indices and coordinate values
    """
    ms_dict = dict()
    index_stop = int(p.getmz(0).shape[0] * 0.999)

    # should pixel 0 be chosen? looks like mz is same across indices
    ms_dict["mz"] = p.getmz(0)[:index_stop]

    ms_dict["I"] = p.get_intensity_chunk(
        chunk_start=chunk_start,
        chunk_end=(chunk_start + chunk_size),
        index_stop=index_stop,
    )
    ms_dict["I"] = ms_dict["I"][:, :index_stop]

    ms_dict["coords"] = np.asarray(
        p.coordinates[chunk_start: (chunk_start + chunk_size)]
    )

    return ms_dict


def get_chunk_ms_info_inhomogeneous(
    p, chunk_start, chunk_size, max_mz, min_mz, mz_RP, RP, dist, rp_factor
):
    """Creates a dictionary of intensity, m/z, and coordinate values from a chunk of imzML data of inhomogeneous shapes

    Parameters
    ----------
    p : pyimzml.ImzMLParser
        Instace of ImzMLParser class containing imported imzML data
    chunk_start : int
        Pixel index for the start of chunk
    chunk_size : int
        Number of pixels contained in each chunk
    max_mz : int
        Upper m/z cutoff of MS analysis
    min_mz : int
        Lower m/z cutoff of MS analysis
    mz_RP : int
        m/z at which resolving power (FWHM) is calculated
    RP : int
        Resolving power in FWHM
    dist : int
        Minimum number of data points required to discern closely-spaced peaks
    rp_factor : int
        Coefficient to scale up or down number of bins

    Returns
    -------
    ms_dict : dict
        Key `I` contains the intensity matrix of axes pixel index and m/z bin index
        Key `mz` contains m/z values
        Key `coords` contains pixel indices and coordinate values
    """
    ms_dict = dict()
    ms_dict["I"] = []
    ms_dict["coords"] = []

    for idx, (x, y, z) in enumerate(
        p.coordinates[chunk_start: chunk_start + chunk_size]
    ):
        bin_spectrum = np.zeros(
            [int((max_mz - min_mz) / (mz_RP / RP) * rp_factor) + 1])
        index = chunk_start + idx
        mzs, intensities = p.getspectrum(index)
        intensity_index = np.round((mzs - min_mz) / (mz_RP / RP) * rp_factor, 0).astype(
            np.int32
        )
        intensity_index[intensity_index == bin_spectrum.shape[0]] -= 1
        bin_spectrum[intensity_index] += intensities
        ms_dict["I"].append(bin_spectrum)
        ms_dict["coords"].append(p.coordinates[index])
    ms_dict["I"] = np.array(ms_dict["I"])
    ms_dict["coords"] = np.array(ms_dict["coords"])

    return ms_dict


def chunk_prep(p, percent_RAM=5):
    """Prepares parameters used in chunking. Imports imzML datasets using pyimzml.ImzMLParser (https://github.com/alexandrovteam/pyimzML)

    Parameters
    ----------
    p :  pyimzml.ImzMLParser
        The data to process, converted to ImzMLParser from the .imzML raw date file.
    percent_RAM : int, optional
        percent available RAM occupied by chunk, by default 5

    Returns
    -------
    num_chunks : int
        Number of chunks the data needs to be devided into to optimize the use of
        avilable RAM on the computer.
    chunk_size_base : int
        The minimal number of columns in each chunk.
    chunk_start : int
        The row index location of where the first chunck starts in the .
    remainder : int
        The remaining columns of data after assigning each chunk a minimal number of columns.

    References
    ----------
    @software{msalign2024,
    author = {Fay, D.; Palmer, A. D.; Vitaly, K.; Alexandrov, T},
    title = {{pyimzML}:A parser for the imzML format used in imaging mass spectrometry.},
    url = {https://github.com/alexandrovteam/pyimzML},
    }
    """
    row_spectra = len(p.intensityLengths)
    col_spectra = p.intensityLengths[0]

    RAM_available = psutil.virtual_memory().available
    num_chunks = (
        int(
            row_spectra
            * col_spectra
            * np.dtype(np.float32).itemsize
            / (RAM_available * percent_RAM / 100)
        )
        + 1
    )

    chunk_size_base = row_spectra // num_chunks

    remainder = row_spectra % num_chunks

    chunk_start = 0

    return num_chunks, chunk_size_base, chunk_start, remainder


def chunk_prep_inhomogeneous(p, n_bins, percent_RAM=5):
    """Prepares parameters used in chunking for imzML datasets of inhomogeneous shapes. Imports imzML datasets using pyimzml.ImzMLParser (https://github.com/alexandrovteam/pyimzML)

    Parameters
    ----------
    p :  pyimzml.ImzMLParser
        The data to process, converted to ImzMLParser from the .imzML raw date file.
    min_mz : float
        minimum m/z in inhomogeneous dataset
    max_mz : float
        maximum m/z in inhomogeneous dataset
    mz_RP : float
        m/z in which resolving power (FWHM) is calculated
    RP : float
        resolving power (FWHM) at m/z {mz_RP}
    dist : int
        minimum number of datapoints for peak separation
    percent_RAM : int, optional
        percent available RAM occupied by chunk, by default 5

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
        Remainder columns after chunk_size_base division

    References
    ----------
    @software{msalign2024,
    author = {Fay, D.; Palmer, A. D.; Vitaly, K.; Alexandrov, T},
    title = {{pyimzML}: A parser for the imzML format used in imaging mass spectrometry.},
    url = {https://github.com/alexandrovteam/pyimzML},
    }
    """
    row_spectra = len(p.coordinates)
    col_spectra = n_bins

    RAM_available = psutil.virtual_memory().available
    num_chunks = (
        int(
            row_spectra
            * col_spectra
            * np.dtype(np.float64).itemsize
            / (RAM_available * percent_RAM / 100)
        )
        + 1
    )

    chunk_size_base = row_spectra // num_chunks

    remainder = row_spectra % num_chunks

    chunk_start = 0

    return num_chunks, chunk_size_base, chunk_start, remainder


def integrate_peak(
    chunk_ms_dict, p2, p2_width, dist, chunk_mz_peaks, integrate_method="peak_width"
):
    """Peak integration based on p2_width over chunk_ns_dict

    Parameters
    ----------
    chunk_ms_dict : dict
        Chunking output from get_chunk_ms_info()
    p2 : array of int
        Peak indices from peak picking
    p2_width : np.1darray
        Peak regions computed by np.peak_widths
    chunk_mz_peaks : np.1darray
        m/z values mapped by p2
    method : str
        Peak integration method. Peak_width is currently the only supported option,
    remove_duplicates : bool, optional
        Removes duplicates of peak regions in cases of near-isobars unresolved at user-specified rel_height in peak_pick(), by default False

    Returns
    -------
    pd.DataFrame
        Integrated intensity matrix
    """

    intensity_df_temp = pd.DataFrame(
        chunk_ms_dict["I"][:, 1:].T, index=chunk_ms_dict["mz"][1:].T
    )
    intensity_df_temp.index.name = "mz"
    intensity_df_temp.reset_index(inplace=True)

    peak_area_df_temp = pd.DataFrame(
        np.zeros(
            (len(p2), intensity_df_temp.shape[1] - 1)
        ),
        index=chunk_mz_peaks,
    )
    if integrate_method == "peak_width":
        peak_width = p2_width[2:].T
        peak_width = np.round(peak_width)

        for column in intensity_df_temp.columns[1:]:
            peak_area = [
                np.trapz(
                    chunk_ms_dict["I"][column][
                        np.arange(
                            np.ceil(row[0]).astype(int), np.ceil(
                                row[1]).astype(int) + 1
                        )
                    ],
                    chunk_ms_dict["mz"][
                        np.arange(
                            np.ceil(row[0]).astype(int), np.ceil(
                                row[1]).astype(int) + 1
                        )
                    ]
                    - chunk_ms_dict["mz"][
                        np.arange(
                            np.ceil(row[0]).astype(int) -
                            1, np.ceil(row[1]).astype(int)
                        )
                    ],
                )
                for row in peak_width
            ]
            peak_area_df_temp.iloc[:, column] = np.asarray(peak_area)
    elif integrate_method == "peak_max":
        for column in intensity_df_temp.columns[1:]:
            peak_area = [
                np.max(
                    chunk_ms_dict["I"][column][
                        np.arange(peak - (dist - 1), peak + (dist - 1))
                    ]
                )
                for peak in p2
            ]
            peak_area_df_temp.iloc[:, column] = np.asarray(peak_area)
    else:
        print("method DNE")

    return peak_area_df_temp


def get_p2(
    p,
    loq,
    lwr,
    upr,
    rel_height,
    chunk_size_base,
    num_chunks,
    chunk_start,
    remainder,
    dist,
    method="point",
):
    """_summary_

    Parameters
    ----------
    p :  pyimzml.ImzMLParser
        The data to process, converted to ImzMLParser from the .imzML raw date file.
    loq : float
        number of times the noise level (k * noise) to define limit of quantification used in peak picking
    lwr : float
        lower m/z bound of a region in spectrum without signals
    upr : float
        upper m/z bound of a region in spectrum without signals
    rel_height : float, optional
        peak height cutoff for peak integration, by default 0.9
    num_chunks: int
        Number of chunks the data needs to be devided into to optimize the use of
        avilable RAM on the computer.
    chunk_size_base: int
        The minimal number of columns in each chunk.
    chunk_start: int
        The row index location of where the first chunck starts in the .
    remainder: int
        Remainder columns after chunk_size_base division
    dist : int
        Minimum number of datapoints for peak separation
    method of computing noise, by default "point"
        method `point` takes specified lower and upper m/z bound of a region in spectrum without signals and compute its standard deviation to define noise level
        method `specify_noise` takes user-specified noise level
        method `automatic` computes standard deviation of non-outliers (< k * z_score), where k is specified by user
        method `binning` re-bins mass spectra (userful for compressed data with inhomogeneous shapes), then computes noise using method "automatic"

    Returns
    -------
    p2 : np.1darray
        m/z bin indices corresponding to peak-picked maxima
    p2_width : np.1darray
        Peak regions computed by np.peak_widths
    noise : float
        Computed noise level used in peak-picking
    mzs : np.1darray
        m/z values element-indexed to m/z bins
    avg_spectrum : np.1darray
        Mean mass spectrum from imzML dataset
    """
    index_stop = int(p.getmz(0).shape[0] * 0.999)
    avg_spectrum = np.zeros(index_stop)
    mzs = p.getmz(0)[:index_stop]
    noise = 0
    for i in range(num_chunks):
        print(f"Chunk {i}")
        start_time = time.time()
        chunk_size_temp = chunk_size_base
        if remainder > i:
            chunk_size_temp += 1

        intensities_chunk = p.get_intensity_chunk(
            chunk_start=chunk_start,
            chunk_end=(chunk_start + chunk_size_temp),
            index_stop=index_stop,
        )
        avg_spectrum += np.sum(intensities_chunk, axis=0)

        if method == "point":
            noise_bin = np.std(
                np.asarray(intensities_chunk)[
                    :, np.logical_and(mzs >= lwr, mzs <= upr)
                ],
                axis=0,
            )
            if np.sum(noise_bin) == 0:
                pass
            else:
                noise += np.mean(noise_bin[noise_bin != 0])

            chunk_start += chunk_size_temp

        print(
            f"Time used for running chunk {i} with a size of \n"
            f"{chunk_size_temp}: {time.time()-start_time}"
        )
        gc.collect()

    avg_spectrum /= len(p.coordinates)

    if method == "point":
        noise /= num_chunks

    p2, info = find_peaks(x=avg_spectrum, height=loq * noise, distance=dist)
    p2_width = peak_widths(avg_spectrum, p2, rel_height=rel_height)
    p2_width = np.asarray(p2_width)

    return p2, p2_width, noise, mzs, avg_spectrum


# ===from transfer_learning_python===#


def remapping_coords(x, y):
    """Remap x,y coordinate values on arbitrary scale to minima (0, 0) with steps of 1

    Parameters
    ----------
    x : np.array
        x coordinate values imported from imzML dataset
    y : np.array
        y coordinate values imported from imzML dataset

    Returns
    -------
    x : np.array
        remapped x coordinate values
    y : np.array
        remapped y coordinate values
    """

    #    x = x.to_numpy()
    #    y = y.to_numpy()
    a, b = 0, 0
    for i in np.sort(np.unique(x)):
        x[x == i] = a
        a += 1
    for j in np.sort(np.unique(y))[::-1]:
        y[y == j] = b
        b += 1

    return x, y


def make_image_1c(data_2darray, remap_coord=True, max_normalize=True):
    """Map pixel-indexed intensity values to x,y-coordinate values. Just-in-time compiler implementation

    Parameters
    ----------
    data_2darray : np.ndarray
        2darray converted from intensity matrix
    img_array_1c : np.ndarray
        empty 3darray to store x,y-mapped ion images

    Returns
    -------
    np.3darray
        RGB-transformed and grayscale images
    """

    x_min = min(data_2darray[:, -2])
    y_min = min(data_2darray[:, -1])

    if remap_coord:
        data_2darray[:, -2] = data_2darray[:, -2] - x_min
        data_2darray[:, -1] = data_2darray[:, -1] - y_min

    img_array_1c = np.zeros(
        [
            data_2darray.shape[1] - 2,
            int(max(data_2darray[:, -1])) + 1,
            int(max(data_2darray[:, -2])) + 1,
            1,
        ]
    )

    if max_normalize:
        data_2darray[:, :-2] = data_2darray[:, :-2] / np.max(
            data_2darray[:, :-2], axis=0
        )

    for k in range(data_2darray.shape[1] - 2):

        for row, coord in enumerate(data_2darray[:, -2:]):
            row = int(row)
            img_array_1c[k, int(coord[1]), int(coord[0]),
                         0:1] = data_2darray[row, k]

    if np.isnan(img_array_1c).any():
        print("Nan exists in single-channel images")

    return img_array_1c


@jit(nopython=True)
def make_image_1c_njit(data_2darray, img_array_1c, remap_coord=True):
    """Map pixel-indexed intensity values to x,y-coordinate values. Just-in-time compiler implementation

    Parameters
    ----------
    data_2darray : np.ndarray
        2darray converted from intensity matrix
    img_array_1c : np.ndarray
        empty array to store x,y-mapped ion images

    Returns
    -------
    img_array_1c : np.ndarray
        RGB-transformed and grayscale images
    """

    x_min = min(data_2darray[:, -2])
    y_min = min(data_2darray[:, -1])

    if remap_coord:
        data_2darray[:, -2] = data_2darray[:, -2] - x_min
        data_2darray[:, -1] = data_2darray[:, -1] - y_min

    for k in range(data_2darray.shape[1] - 2):

        for row, coord in enumerate(data_2darray[:, -2:]):
            row = int(row)
            img_array_1c[k, int(coord[1]), int(coord[0]),
                         0:1] = data_2darray[row, k]

    if np.isnan(img_array_1c).any():
        pass

    return img_array_1c


def max_normalize(data_2darray):
    """Normalize on maximum value of each pixel

    Parameters
    ----------
    data_2darray : np.ndarray
        ndarray converted from intensity matrix extracted from imzML dataset

    Returns
    -------
    data_2darray : np.ndarray
        Max-normalized 2darray from intensity matrix extracted from imzML dataset
    """
    data_2darray = data_2darray / np.max(data_2darray, axis=0)
    return data_2darray


def draw_ROI(
    ROI_info, show_ROI=True, show_square=False, linewidth=3, ROI_size_divisor=8
):
    """Visualize selected ROI region in a green box

    Parameters
    ----------
    ROI_info : pd.dataframe
        Replicate, ROI annotations, x,y coordinates from ROI selection
    show_ROI : bool, optional
        Displays ROI labels if True, by default True
    show_square : bool, optional
        Displays green box around selected ROI if True , by default False
    linewidth : float, optional
        Width of green box, by default 3
    ROI_size_divisor : float, optional
            Controls size of ROI labels, where a smaller divisor gives a larger ROI label, by default 8
    """
    start_col = np.where(ROI_info.columns == "bottom")[0][0]
    truncate_col = np.where(ROI_info.columns == "right")[0][0]
    squares = ROI_info.iloc[:, start_col: truncate_col + 1].to_numpy()

    for i, square in enumerate(squares):
        bottom, top, left, right = square
        # plt.ylim(top-10, bottom+10)
        # plt.xlim(left-10, right+10)

        x_coords = [left, left, right, right, left]
        y_coords = [bottom, top, top, bottom, bottom]
        if show_square:
            plt.plot(x_coords, y_coords, "g-", linewidth=linewidth)
        if show_ROI:
            plt.text(
                (left + right) / 2,
                bottom - 3,
                ROI_info["ROI"][i],
                horizontalalignment="center",
                size=max(ROI_info["right"]) / ROI_size_divisor,
                color="white",
            )


def plot_image_at_point(im, xy, zoom, color_scheme="inferno"):
    """
    Plots a tiny image at point xy for visualization with dimensionally reduced embedding.
    """

    # dxy = np.random.rand(int(np.floor(dxy)))/50 * plt.ylim()
    # plt.arrow(*xy, *dxy)
    ab = AnnotationBbox(
        OffsetImage(im, zoom=zoom, cmap=color_scheme), xy, frameon=False
    )
    plt.gca().add_artist(ab)


def get_spectrum(output_filepath, mzs, avg_intensity, p2):
    """Generate interactive mean spectrum and stores at output_filepath

    Parameters
    ----------
    output_filepath : str
        file path to store spectrum
    mzs : np.array
        m/z values from get_p2
    avg_intensity : np.array
        mean spectrum from get_p2
    p2 : np.array
        m/z bin indices corresponding to peak-picked maxima
    """
    output_file(filename=output_filepath)

    ms_spectrum = figure(width=1400, height=600,
                         title="Interactive mass spectrum")
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


def repel_labels(ax, x, y, labels, colors, k=0.01, jitter_factor=100, font_size=5):
    """Codes refactored from (https://stackoverflow.com/questions/34693991/repel-annotations-in-matplotlib)"""
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    i = 0
    for xi, yi, label in zip(x, y, labels):
        data_str = f"data_{i}_label_{label}"
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)
        i += 1

    pos = nx.spring_layout(
        G, pos=init_pos, fixed=data_nodes, k=k * jitter_factor)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
    scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)

    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val * scale) + shift

    for (label, data_str), color in zip(G.edges(), colors):
        # displacement = (2*np.exp(np.random.rand(2)) - np.exp(1)) * k
        # displacement[1] *= 10
        ax.annotate(
            label,
            xy=pos[data_str],
            xycoords="data",
            xytext=pos[label],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3",
                color=color,
                alpha=0.3,
            ),
            fontsize=font_size,
            color=color,
            alpha=0.5,
        )
    # expand limits
    all_pos = np.vstack(list(pos.values()))
    x_span, y_span = np.ptp(all_pos, axis=0)
    mins = np.min(all_pos - x_span * 0.15, 0)
    maxs = np.max(all_pos + y_span * 0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])


def str2bool(v):
    """Converts `yes` to True"""
    return v.lower() in ("yes")


@jit(nopython=True)
def get_MS1_hits(mz, MS1_ion_mass, ppm_threshold):
    """MS1 search within a ppm threshold

    Parameters
    ----------
    mz : np.array or np.ndarray
        Experimental m/z
    MS1_ion_mass : np.array
        m/z array of ions in MS1 database
    ppm_threshold : float
        ppm threshold for MS1 hits

    Returns
    -------
    MS1_hit_indices : np.array or np.ndarray
        MS1 hits indexed by positions of `mz`
    ppm : np.array or np.ndarray
        ppm values of each pair of experimental m/z and theoretical m/z from MS1 database
    """
    MS1_hits = np.abs((mz - MS1_ion_mass) / mz * 10**6) < ppm_threshold
    MS1_hit_indices = np.where(MS1_hits)
    ppm = np.abs((mz - MS1_ion_mass) / mz * 10**6)
    return MS1_hit_indices, ppm


# starting here, codes copied or refactored from https://github.com/lukasz-migas/msalign
def format_time(value: float) -> str:
    """Convert time to nicer format. Codes from https://github.com/lukasz-migas/msalign"""
    if value <= 0.005:
        return f"{value * 1000000:.0f}us"
    elif value <= 0.1:
        return f"{value * 1000:.1f}ms"
    elif value > 86400:
        return f"{value / 86400:.2f}day"
    elif value > 1800:
        return f"{value / 3600:.2f}hr"
    elif value > 60:
        return f"{value / 60:.2f}min"
    return f"{value:.2f}s"


def time_loop(t_start: float, n_item: int, n_total: int, as_percentage: bool = True) -> str:
    """Calculate average, remaining and total times. Codes from https://github.com/lukasz-migas/msalign

    Parameters
    ----------
    t_start : float
        starting time of the for loop
    n_item : int
        index of the current item - assumes index starts at 0
    n_total : int
        total number of items in the for loop - assumes index starts at 0
    as_percentage : bool, optional
        if 'True', progress will be displayed as percentage rather than the raw value

    Returns
    -------
    timed : str
        loop timing information
    """
    t_tot = time.time() - t_start
    t_avg = t_tot / (n_item + 1)
    t_rem = t_avg * (n_total - n_item + 1)

    # calculate progress
    progress = f"{n_item}/{n_total + 1}"
    if as_percentage:
        progress = f"{(n_item / (n_total + 1)) * 100:.1f}%"

    return f"[Avg: {format_time(t_avg)} | Rem: {format_time(t_rem)} | Tot: {format_time(t_tot)} || {progress}]"


def shift(array, num, fill_value=0):
    """Shift 1d array to new position with 0 padding to prevent wraparound - this function is actually
    quicker than np.roll. Codes from https://github.com/lukasz-migas/msalign

    Parameters
    ----------
    array : np.ndarray
        array to be shifted
    num : int
        value by which the array should be shifted
    fill_value : Union[float, int]
        value to fill in the areas where wraparound would have happened
    """
    result = np.empty_like(array)
    if not isinstance(num, int):
        raise ValueError("`num` must be an integer")

    if num > 0:
        result[:num] = fill_value
        result[num:] = array[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = array[-num:]
    else:
        result[:] = array
    return result


def check_xy(x, array):
    """
    Check zvals input. Codes from https://github.com/lukasz-migas/msalign

    Parameters
    ----------
    x : np.ndarray
        1D array of separation units (N). The number of elements of xvals must equal the number of elements of
        zvals.shape[1]
    array : np.ndarray
        2D array of intensities that must have common separation units (M x N) where M is the number of vectors
        and N is number of points in the vector

    Returns
    -------
    zvals : np.ndarray
        2D array that should match the dimensions of xvals input
    """
    if x.shape[0] != array.shape[1]:
        if x.shape[0] != array.shape[0]:
            raise ValueError("Dimensions mismatch")
        array = array.T
        warnings.warn(
            "The input array was rotated to match the x-axis input", UserWarning)

    return array


def generate_function(method, x, y):
    """
    Generate interpolation function. Codes from https://github.com/lukasz-migas/msalign

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
    Creates an interpolator function for given control points and their values. Codes from https://github.com/lukasz-migas/msalign

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


def find_nearest_index(x: np.ndarray, value: Union[float, int]):
    """Find index of nearest value, Codes from https://github.com/lukasz-migas/msalign

    Parameters
    ----------
    x : np.array
        input array
    value : number (float, int)
        input value
    Returns
    -------
    index : int
        index of nearest value
    """
    x = np.asarray(x)
    return np.argmin(np.abs(x - value))


def convert_peak_values_to_index(x: np.ndarray, peaks) -> List:
    """Converts non-integer peak values to index value by finding
    the nearest value in the `xvals` array. Codes from https://github.com/lukasz-migas/msalign

    Parameters
    ----------
    x : np.array
        input array
    peaks : list
        list of peaks

    Returns
    -------
    peaks_idx : list
        list of peaks as index
    """
    return [find_nearest_index(x, peak) for peak in peaks]


def find_nearest_index_gpu(x: cp.ndarray, value: Union[float, int]):
    """Find index of nearest value. Codes refactored from https://github.com/lukasz-migas/msalign

    Parameters
    ----------
    x : np.array
        input array
    value : number (float, int)
        input value
    Returns
    -------
    index : int
        index of nearest value
    """
    x = cp.asarray(x)
    return cp.argmin(cp.abs(x - value))


def convert_peak_values_to_index_gpu(x: cp.ndarray, peaks) -> List:
    """Converts non-integer peak values to index value by finding
    the nearest value in the `xvals` array. Codes refactored from https://github.com/lukasz-migas/msalign

    Parameters
    ----------
    x : np.array
        input array
    peaks : list
        list of peaks

    Returns
    -------
    peaks_idx : list
        list of peaks as index
    """
    return [find_nearest_index_gpu(x, peak) for peak in peaks]


try:
    import cupy as cp

    print(f"cupy is installed and imported successfully!")
except ImportError:
    print(f"cupy is not installed or could not be imported.")

METHODS = ["pchip", "zero", "slinear",
           "quadratic", "cubic", "linear", "gpu_linear"]
LOGGER = logging.getLogger(__name__)


class Aligner_CPU:
    """Alignment class from https://github.com/lukasz-migas/msalign

    References
    ----------
        @software{msalign2024,
        author = {Lukasz G. Migas},
        title = {{msalign}: Spectral alignment based on MATLAB's `msalign` function.},
        url = {https://github.com/lukasz-migas/msalign},
        version = {0.2.0},
        year = {2024},
        }
    """

    _method, _gaussian_ratio, _gaussian_resolution, _gaussian_width, _n_iterations = (
        None,
        None,
        None,
        None,
        None,
    )
    _corr_sig_l, _corr_sig_x, _corr_sig_y, _reduce_range_factor, _scale_range = (
        None,
        None,
        None,
        None,
        None,
    )
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
            LOGGER.warning(
                "Only computing shifts - changed `align_by_index` to `True`."
            )

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
            raise ValueError(
                f"Method `{value}` not found in the method options: {METHODS}"
            )
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
            raise ValueError(
                "Value of 'iterations' must be above 0 and be an integer!")
        self._n_iterations = value

    @property
    def grid_steps(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._grid_steps

    @grid_steps.setter
    def grid_steps(self, value: int):
        if value < 1 or not isinstance(value, int):
            raise ValueError(
                "Value of 'iterations' must be above 0 and be an integer!")
        self._grid_steps = value

    @property
    def shift_range(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._shift_range

    @shift_range.setter
    def shift_range(self, value: ty.Tuple[float, float]):
        if len(value) != 2:
            raise ValueError(
                "Number of 'shift_values' is not correct. Shift range accepts"
                " numpy array with two values."
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
            raise ValueError(
                "Number of weights does not match the number of peaks.")
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
            corr_sig_x[:, i] = left_l + (
                gaussian_resolution_range
                * (right_l - left_l)
                / self.gaussian_resolution
            )
            corr_sig_y[:, i] = self.weights[i] * np.exp(
                -np.square(
                    (corr_sig_x[:, i] - self.peaks[i]) / gaussian_widths[i]
                )  # noqa
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
            np.vstack([mesh_a.flatten(order="F"),
                      mesh_b.flatten(order="F")]).T,
            [1, self._n_iterations],
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
            self.shift_opt[n_signal], self.scale_opt[n_signal] = self.compute(
                y)
        LOGGER.debug(
            f"Processed {self.n_signals} signals "
            + time_loop(t_start, self.n_signals + 1, self.n_signals)
        )
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
            scale_grid = _scale[0] + self._search_space[:, (n_iter * 2) - 2] * np.diff(
                _scale
            )
            shift_grid = _shift[0] + self._search_space[:, (n_iter * 2) + 1] * np.diff(
                _shift
            )
            temp = (
                np.reshape(scale_grid, (scale_grid.shape[0], 1))
                * np.reshape(self._corr_sig_x, (1, self._corr_sig_l))
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
            _scale = (
                scale_opt + _scale_range *
                np.diff(_scale) * self._reduce_range_factor
            )
            _shift = (
                shift_opt + _scale_range *
                np.diff(_shift) * self._reduce_range_factor
            )
        return shift_opt, scale_opt

    def apply(self, return_shifts: bool = None):
        """Align the signals against the computed values"""
        if not self._computed:
            warnings.warn(
                "Aligning data without computing optimal alignment parameters",
                UserWarning,
            )
        self._return_shifts = (
            return_shifts if return_shifts is not None else self._return_shifts
        )

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
            self.array_aligned[iteration] = self._apply(
                y, shift_opt[iteration], scale_opt[iteration]
            )
        self.shift_values = self.shift_opt

        LOGGER.debug(
            f"Re-aligned {self.n_signals} signals "
            + time_loop(t_start, self.n_signals + 1, self.n_signals)
        )

    def _apply(self, y: np.ndarray, shift_value: float, scale_value: float):
        """Apply alignment correction to array `y`."""
        func = generate_function(
            self.method, (self.x - shift_value) / scale_value, y)
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
            self.array_aligned[iteration] = self._shift(
                y, shift_opt[iteration])
        self.shift_values = shift_opt

        LOGGER.debug(
            f"Re-aligned {self.n_signals} signals "
            + time_loop(t_start, self.n_signals + 1, self.n_signals)
        )

    @staticmethod
    def _shift(y: np.ndarray, shift_value: float):
        """Apply shift correction to array `y`."""
        return shift(y, -int(shift_value))


class Aligner_GPU:
    """Alignment class refactored from https://github.com/lukasz-migas/msalign

    Cite the following if used:
        @software{msalign2024,
        author = {Lukasz G. Migas},
        title = {{msalign}: Spectral alignment based on MATLAB's `msalign` function.},
        url = {https://github.com/lukasz-migas/msalign},
        version = {0.2.0},
        year = {2024},
        }
    """

    def __init__(
        self,
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
        align_by_index: bool = False,
    ):

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
            LOGGER.warning(
                "Only computing shifts - changed `align_by_index` to `True`."
            )

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
            cp.vstack([mesh_a.flatten(order="F"),
                      mesh_b.flatten(order="F")]).T,
            [1, self._n_iterations],
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
            raise ValueError(
                f"Method `{value}` not found in the method options: {METHODS}"
            )
        self._method = value

    @property
    def grid_steps(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._grid_steps

    @grid_steps.setter
    def grid_steps(self, value: int):
        if value < 1 or not isinstance(value, int):
            raise ValueError(
                "Value of 'iterations' must be above 0 and be an integer!")
        self._grid_steps = value

    @property
    def n_iterations(self):
        """Total number of iterations - increase to improve accuracy."""
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, value: int):
        if value < 1 or not isinstance(value, int):
            raise ValueError(
                "Value of 'iterations' must be above 0 and be an integer!")
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
            self.shift_opt[n_signal], self.scale_opt[n_signal] = self.compute(
                y)
        LOGGER.debug(
            f"Processed {self.n_signals} signals "
            + time_loop(t_start, self.n_signals + 1, self.n_signals)
        )
        self._computed = True

    def run_batch(self, n_iterations: int = None):
        """Execute the alignment procedure for the entire batch of signals and collate the shift/scale vectors."""
        self.n_iterations = n_iterations or self.n_iterations
        t_start = time.time()

        # Process all signals at once using the modified compute function
        self.shift_opt, self.scale_opt = self.compute_batch(self.array)

        # Here, use appropriate logging to account for CuPy/GPU execution if needed
        print(
            f"Processed {self.array.shape[0]} signals in \n"
            f"{time.time() - t_start} seconds"
        )
        self._computed = True

    def compute_batch(self, Y: cp.ndarray) -> ty.Tuple[cp.ndarray, cp.ndarray]:
        """
        Compute correction factors for a batch of signals.
        """
        _scale_range = cp.array([-0.5, 0.5])

        # Initialize arrays to hold the optimal scale and shift for each signal
        scale_opts = cp.zeros(Y.shape[0], dtype=cp.float32)
        shift_opts = cp.ones(Y.shape[0], dtype=cp.float32)

        _shift = self.shift_range.copy()
        _scale = self._scale_range.copy()

        funcs = generate_function(self.method, self.x, Y)

        for n_iter in range(self.n_iterations):
            scale_grid = _scale[0] + self._search_space[:, (n_iter * 2) - 2] * cp.diff(
                _scale
            )
            shift_grid = _shift[0] + self._search_space[:, (n_iter * 2) + 1] * cp.diff(
                _shift
            )

            temp = (
                cp.reshape(scale_grid, (-1, 1, 1)) * self._corr_sig_x
                + shift_grid[:, None, :]
            )
            temp = temp.reshape(temp.shape[0], -1)
            temp = cp.nan_to_num(funcs(temp).reshape(
                Y.shape[0], len(scale_grid), -1))

            # Find the best position for each signal
            i_max = cp.argmax(cp.dot(temp, self._corr_sig_y.T), axis=1)

            # Update optimum values
            scale_opts = scale_grid[i_max]
            shift_opts = shift_grid[i_max]

            # Readjust grid for next iteration
            _scale = (
                scale_opts[:, None]
                + _scale_range * cp.diff(_scale) * self._reduce_range_factor
            )
            _shift = (
                shift_opts[:, None]
                + _scale_range * cp.diff(_shift) * self._reduce_range_factor
            )

        return shift_opts, scale_opts

    def apply(self, return_shifts: bool = None):
        """Align the signals against the computed values"""
        if not self._computed:
            warnings.warn(
                "Aligning data without computing optimal alignment parameters",
                UserWarning,
            )
        self._return_shifts = (
            return_shifts if return_shifts is not None else self._return_shifts
        )

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
            self.array_aligned[iteration] = self._apply(
                y, shift_opt[iteration], scale_opt[iteration]
            )
        self.shift_values = self.shift_opt

    def _apply(self, y: cp.ndarray, shift_value: float, scale_value: float):
        """Apply alignment correction to array `y`."""
        func = generate_function(
            self.method, (self.x - shift_value) / scale_value, y)
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
            scale_grid = _scale[0] + self._search_space[:, (n_iter * 2) - 2] * cp.diff(
                _scale
            )
            shift_grid = _shift[0] + self._search_space[:, (n_iter * 2) + 1] * cp.diff(
                _shift
            )
            temp = (
                cp.reshape(scale_grid, (scale_grid.shape[0], 1))
                * cp.reshape(self._corr_sig_x, (1, self._corr_sig_l))
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
            _scale = (
                scale_opt + _scale_range *
                cp.diff(_scale) * self._reduce_range_factor
            )
            _shift = (
                shift_opt + _scale_range *
                cp.diff(_shift) * self._reduce_range_factor
            )
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
            corr_sig_x[:, i] = left_l + (
                gaussian_resolution_range
                * (right_l - left_l)
                / self.gaussian_resolution
            )
            corr_sig_y[:, i] = self.weights[i] * cp.exp(
                -cp.square(
                    (corr_sig_x[:, i] - self.peaks[i]) / gaussian_widths[i]
                )  # noqa
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
            cp.vstack([mesh_a.flatten(order="F"),
                      mesh_b.flatten(order="F")]).T,
            [1, self._n_iterations],
        )

    def shift(self, shift_opt=None):
        """Quickly shift array based on the optimized shift parameters.

        This method does not interpolate but rather moves the data left and right without applying any scaling.

        Parameters
        ----------
        shift_opt: Optional[cp.ndarray]
            vector containing values by which to shift the array
        """
        t_start = time.time()
        if shift_opt is None:
            shift_opt = cp.round(self.shift_opt).astype(cp.int32)

        # quickly shift based on provided values
        for iteration, y in enumerate(self.array):
            self.array_aligned[iteration] = self._shift(
                y, shift_opt[iteration])
        self.shift_values = shift_opt

        LOGGER.debug(
            f"Re-aligned {self.n_signals} signals "
            + time_loop(t_start, self.n_signals + 1, self.n_signals)
        )

    @staticmethod
    def _shift(y: cp.ndarray, shift_value: float):
        """Apply shift correction to array `y`."""
        return shift(y, -int(shift_value))
