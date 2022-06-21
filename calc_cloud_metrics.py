#
# code to calculate cloud metrics based on clear-sky radiation calculated
# based on matlab code calc_cloud_metrics_final.m
# 
# # with solmod2Djc_calc_clear_shade.m
# # author Jono Conway
# jono.conway@niwa.co.n
#

import datetime as dt
import numpy as np
import pandas as pd
from pvlib import irradiance
from pvlib.atmosphere import alt2pres

sigma = 5.67e-8


def k_from_vp(vp):
    """
    calculates shortwave extinction coefficient based on vapour pressure
    :param vp:
    :return:
    """
    k = 0.1715 + 0.07182 * vp
    k[k > 0.95] = 0.95
    return k


def neff_from_trc(trc, vp=None, k=0.65, limit=True):
    # shortwave effective cloudiness
    # variable k based on vp

    if vp is not None:  # calculate k based on Conway et al 2016 IJC
        k = k_from_vp(vp)

    # hourly values
    neff2 = (1 - trc) / k

    if limit:
        neff2[neff2 < 0] = 0
        neff2[neff2 > 1] = 1

    return neff2


def dtrc_from_sw(swin, sw_pot, blocksize, filter=None,keep_nan=True):
    # shortwave transmission using daily sums of swin and clear-sky sw
    # blocksize must be specified
    # filter  removes values when shaded
    # no limits imposed

    d_swinsum = blockfun(swin, blocksize, filter, method='sum',keep_nan=True)
    d_swpotsum = blockfun(sw_pot, blocksize, filter, method='sum',keep_nan=True)

    d_trc = d_swinsum / d_swpotsum

    return d_trc


def neff_sum24_from_sw(swin, sw_pot, blocksize, filter=None, vp=None, dk=0.65, limit=True):
    # shortwave effective cloudiness using daily sums of swin and clear-sky sw
    # blocksize must be specified
    # filter  removes values when shaded
    # returns same values for each hour of the day

    d_trc = dtrc_from_sw(swin, sw_pot, blocksize, filter)

    if vp is not None:
        d_vp = blockfun(vp, blocksize, filter)  # average vapour pressure over daytime
        dk = k_from_vp(d_vp)

    d_neff5 = (1 - d_trc) / dk

    if limit:
        d_neff5[d_neff5 < 0] = 0
        d_neff5[d_neff5 > 1] = 1

    neff5 = blockfun(d_neff5, -1 * blocksize)

    return neff5


def ecs_from_ea_ta(vp, tc2, ead=0.22, b=0.456, m=7):
    # calculate clear-sky emissivity according to Konzelman et al. using coefficients based on Durr 2006, including resetting m to 7
    ecs = ead + b * (vp * 100 / (tc2 + 273.15)) ** (1 / m)
    return ecs


def eeff_from_lw_tk(lwin, tk, svf=1):
    if svf < 1:
        lwin_open = cor_lw_for_terrain(lwin, tk, svf)
    else:
        lwin_open = lwin
    eeff = lwin_open / (sigma * tk ** 4)
    return eeff


def cor_lw_for_terrain(lwin, tk, svf=1):
    # calculate incoming longwave radiation without influence of terrain
    # assumes terrain is a air temperature (an ok first assumption as the increase in emissivitiy from clear-sky is more important than radiant temperature)
    lwter = (1 - svf) * sigma * tk ** 4
    # incoming longwave with terrain component removed
    lwin_open = (lwin - lwter) / svf
    return lwin_open


def nep_from_ecs(eeff, ecs, limit=True):
    # calculate longwave effective cloudiness
    nep1 = (eeff - ecs) / (1 - ecs)
    if limit:
        nep1[nep1 < 0] = 0
        nep1[nep1 > 1] = 1
    return nep1


def ea_from_tc_rh(tc, rh, pres_hpa=None):
    # vapour pressure in hPa calculated according to Buck
    # Rh should be with respect to water, not ice.
    if pres_hpa == None:
        ea = 6.1121 * np.exp(17.502 * tc / (240.97 + tc)) * rh / 100
    else:
        ea = (1.0007 + (3.46 * 1e-6 * pres_hpa)) * 6.1121 * np.exp(17.502 * tc / (240.97 + tc)) * rh / 100
    return ea


def w_from_ea_tk(ea, tk):
    # precipitable water according to Prata 1996
    w = (46.5 * ea / (tk))  # prata 1996
    return w


def find_cs_from_lw(ldr, ea, tk):
    # identify periods that are in the lowest 10% of obs for given bin of ea/tk, with which to calibrate lw cs model
    ea_tk = ea / tk
    # split data into 30 bins along ea_tk, then find all the points within the lowest 10% of each bin.
    bin_edges = np.linspace(np.min(ea_tk), np.max(ea_tk), 30)
    ind = np.full(ldr.shape, False, dtype=np.bool)  # set up empty array
    for i in range(29):
        test_ind = np.logical_and(ea_tk >= bin_edges[i], ea_tk < bin_edges[i + 1])
        if np.sum(test_ind) == 0:
            print('skipping bin')
        else:
            # define range of points to test according to ea/tk bin
            p5 = np.percentile(ldr[test_ind], 10)  # find 10th percentile of ldr
            ind[np.logical_and(test_ind, ldr < p5)] = True  # activate these in the filter.

    return ind


def optimise_lw_cs(ldr, ea, tk, svf=1, alt=None, m=7, return_stats=False):
    # optimise lw cs model using sub set of data
    if alt is not None:
        # alt = np.asarray([490, 1610, 2690, 3580])
        # ead = np.asarray([0.23, 0.22, 0.21, 0.20])
        # coef = np.polyfit(alt, ead, 1)
        p = np.poly1d([-9.63601916e-06, 2.35163370e-01])
        ead = p(alt)
    else:
        ead = 0.23

    ind10 = find_cs_from_lw(ldr, ea, tk)
    ea10 = ea[ind10]
    tc10 = tk[ind10] - 273.15
    eeff = eeff_from_lw_tk(ldr[ind10], tk[ind10], svf=svf)
    stat = np.full(301, np.nan)
    b_test = np.linspace(0.25, 0.55, 301)
    for i, b in enumerate(b_test):
        ecs = ecs_from_ea_ta(ea10, tc10, ead=ead, b=b, m=m)
        stat[i] = rmsd(ecs, eeff)
    if return_stats:
        return np.round(b_test[np.argmin(stat)], 3), np.min(stat), ead
    else:
        return np.round(b_test[np.argmin(stat)], 3)


def add_clear_sky(aws_df, aws_loc, method='pvlib', aod700=0.10, k_aes=0.987, oz=0.3, alphss=0.9, albsur=0.45):
    """
    # calculate clear-sky radiation at timestep of index in the dataframe. utc and stime moved to mid point of hourly period in process timezones.
    :param aws_df: dictionary containing data for site
    :param aws_loc: pvlibs Location object
    :param aod700: aerosol optical depth at 700nm
    :param w: percipitable water in cm
    :param method: which method to use. option 'pvlib'
    :return: adds clear-sky global horizontal, diffuse horizontal and direct normal surface radiation to aws dataframe
    """
    # calculate radiation at mid point of hourly period. i.e. 30 min before measurement timestamp.
    # datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
    # times2 = pd.DatetimeIndex(aws_df.utc_time) - dt.timedelta(minutes=datastep / 2)
    #
    times2 = pd.DatetimeIndex(aws_df.utc_time)
    if method == 'pvlib':
        cs = aws_loc.get_clearsky(times2, model='simplified_solis', aod700=aod700, precipitable_water=aws_df.w.values)  # NOTE w is limited a minimum of 0.2
        aws_df['sw_cs_ghi'] = cs.ghi.values
        aws_df['sw_cs_dhi'] = cs.dhi.values
        aws_df['sw_cs_dni'] = cs.dni.values
    else:
        print('incorrect method chosen for clear-sky SW radiation, not calculated')

    return aws_df


def process_timezones(aws_df, lon, datastep):
    if datastep != 60:
        # resample to 1 hour, using
        aws_df = aws_df.resample('1H', label='left', closed='right').mean() # puts average at the start of the time range
        aws_df.index = aws_df.index + dt.timedelta(seconds=1800) # move from the start of the hour to halfway through the hour
        datastep = 60
    else:
        aws_df.index = aws_df.index - dt.timedelta(seconds=1800) # move to halfway through the hour (assuming the timestep was at the end of the hour)
    pydt = aws_df.index.to_pydatetime()
    aws_df['loc_time'] = pydt
    aws_df['utc_time'] = aws_df.index.tz_convert(None)
    # find if timezones has partical hour offset and set up to move to hour when calculating stime
    hours_offset = pydt[0].utcoffset() / dt.timedelta(0, 3600)
    move_time = np.round(hours_offset) - hours_offset
    stime_utc_offset = (np.round(lon / 15) - move_time) * 3600  # get offset from utc using longitude in seconds, rounding to nearest hour
    aws_df['stime'] = aws_df.index.tz_convert(tz=dt.timezone(dt.timedelta(seconds=stime_utc_offset))).tz_localize(
        None)  # convert to local solar time and remove timezone info
    aws_df.index = pd.DatetimeIndex(aws_df['stime'])
    return aws_df


def blockfun(datain, blocksize, filter=None, method='mean', keep_nan=False):
    """
    function to block average data using nan mean. can filter out bad values, and can upsample data. use negative blocksize to upsample data.
    :param datain_copy: array with data to average. data is avearged along First dimension
    :param blocksize: interger specifying the number of elements to average. if positive, must be a multiple of the length of datain. if negative, data is upsampled.
    :param filter: values to filter out. must have same length as first dimension of datain. all values with filter == True are set to nan
    :param method 'mean' or 'sum'
    :param keep_nan True = preserves nans in input, so will return nan if any nan's within the block.
    :return:
    """

    # datain = np.ones([192,2])
    # blocksize = 96
    datain_copy = datain.copy()
    #
    #     if datain_copy.ndim > 1:
    #         datain_copy[filter, :] = np.nan
    #     else:
    #         datain_copy[filter] = np.nan
    filter_array = np.full(datain_copy.shape, True)
    if filter is not None:
        if datain_copy.ndim > 1:
            filter_array[filter, :] = False
        else:
            filter_array[filter] = False

    num_rows = datain_copy.shape[0]

    if blocksize > 0:  # downsample
        if datain_copy.ndim > 1:
            num_col = datain_copy.shape[1]
            dataout = np.ones([int(num_rows / blocksize), num_col])
            for ii, i in enumerate(range(0, num_rows, blocksize)):
                for j in range(num_col):
                    block = datain_copy[i:i + blocksize, j]
                    block = block[filter_array[i:i + blocksize, j]]
                    if method == 'mean':
                        if keep_nan:
                            dataout[ii, j] = np.mean(block)
                        else:
                            dataout[ii, j] = np.nanmean(block)
                    elif method == 'sum':
                        if keep_nan:
                            dataout[ii, j] = np.sum(block)
                        else:
                            dataout[ii, j] = np.nansum(block)
        else:
            dataout = np.ones([int(num_rows / blocksize), ])
            for ii, i in enumerate(range(0, num_rows, blocksize)):
                block = datain_copy[i:i + blocksize]
                block = block[filter_array[i:i + blocksize]]
                if method == 'mean':
                    if keep_nan:
                        dataout[ii] = np.mean(block)
                    else:
                        dataout[ii] = np.nanmean(block)

                elif method == 'sum':
                    if keep_nan:
                        dataout[ii] = np.sum(block)
                    else:
                        dataout[ii] = np.nansum(block)

    elif blocksize < 0:  # upsample ignoring the filter

        blocksize_copy = blocksize * -1

        if datain_copy.ndim > 1:
            dataout = np.ones([int(num_rows * blocksize_copy), datain_copy.shape[1]])
            for i in range(num_rows):
                if method == 'mean':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy, :] = datain_copy[i, :]
                elif method == 'sum':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy, :] = datain_copy[i, :] / blocksize_copy
        else:
            dataout = np.ones([int(num_rows * blocksize_copy), ])
            for i in range(num_rows):
                if method == 'mean':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy] = datain_copy[i]
                if method == 'sum':
                    dataout[i * blocksize_copy: i * blocksize_copy + blocksize_copy] = datain_copy[i] / blocksize_copy

    return dataout


def rmsd(y_sim, y_obs):
    """
    calculate the mean bias difference (taken from Ayala, 2017, WRR)

    :param y_sim: series of simulated values
    :param y_obs: series of observed values
    :return:
    """
    assert y_sim.ndim == 1 and y_obs.ndim == 1 and len(y_sim) == len(y_obs)

    rs = np.sqrt(np.mean((y_sim - y_obs) ** 2))

    return rs