# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:30:43 2024

@author: yutin
"""


import numba
from numba import cuda


import os
import cupy as cp


# peak_integration(column, intensity_df_temp, deviation_df_temp,
#                  chunk_0_mz, p2_new_temp, peak_area_df_temp, chunk_0_mz_peaks)  # axis 1 to colunmn operate
# # peak_area_df is updated in the function



def peak_integration(column, intensity_df, deviation_df, MZ, p2_new, peak_area_df, chunk_0_mz):
    count_rep = 3
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
                peak_area = (cp.trapz(intensity_df[column][value_left:value], MZ[value_left:value]) +
                             cp.trapz(intensity_df[column][value:value_right-1], MZ[value:value_right-1]) +
                             cp.trapz(intensity_df[column][value_left:(value_left+extra_bins+1)], MZ[value_left:(value_left+extra_bins+1)]))
                # peak_area_df.at[MZ[value], 'left_width'] += MZ[value_left]
                # peak_area_df.at[MZ[value], 'right_width'] += MZ[value_right-1]
            elif extra_bins < 0:
                peak_area = (cp.trapz(intensity_df[column][value_left+1:value], MZ[value_left+1:value]) +
                             cp.trapz(intensity_df[column][value:value_right], MZ[value:value_right]) +
                             cp.trapz(intensity_df[column][(value_right+extra_bins-1):value_right], MZ[(value_right+extra_bins-1):value_right]))
                # peak_area_df.at[MZ[value], 'left_width'] += MZ[value_left+1]
                # peak_area_df.at[MZ[value], 'right_width'] += MZ[value_right]
            else:
                peak_area = (cp.trapz(intensity_df[column][value_left:value], MZ[value_left:value]) +
                             cp.trapz(intensity_df[column][value:value_right], MZ[value:value_right]))
                # peak_area_df.at[MZ[value], 'left_width'] += MZ[value_left]
                # peak_area_df.at[MZ[value], 'right_width'] += MZ[value_right]
            peak_area_df[k, column] = peak_area
            k += 1