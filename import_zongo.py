"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib
from calc_cloud_metrics import blockfun

zongo = Location(-16.25, -68.167, tz='America/La_Paz', altitude=5040, name='Zongo Glacier')

flux_input_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Zongo/updated fluxes Sept 2020/input.csv"
flux_output_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Zongo/updated fluxes Sept 2020/output-aws.csv"
outfile_aws_update = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/zongo_aws_with_qm.pkl'
flux_output = pd.read_csv(flux_output_file, skiprows=1)
flux_output.index = pd.date_range(start='1999-09-02 01:00', end='2000-08-30 00:00', freq='1H', tz=zongo.tz)
flux_input = pd.read_csv(flux_input_file, skiprows=1)
flux_input.index = pd.date_range(start='1999-09-01 00:00', end='2000-08-30 23:00', freq='1H', tz=zongo.tz)
full_pd = pd.merge(flux_input, flux_output, how='outer', left_index=True, right_index=True, suffixes=['_input', '_output'])
full_pd.to_pickle(outfile_aws_update)

zongo_file_flux = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Zongo/flux-turb-zgo-9900-hourly.csv"
zongo_file_AWS = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Zongo/1999_2000_OnGlacierData.csv"
zongo_file_albedo = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Zongo/zgo-9900-albedo.csv"
outfile_aws = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/zongo_aws.pkl'

# import AWS file
dat = np.genfromtxt(zongo_file_AWS, delimiter=',', skip_header=3)  # ignore first column as time
headers = np.genfromtxt(zongo_file_AWS, delimiter=',', skip_header=2, skip_footer=8760, dtype=(str))
# put into dictionary
zongo_aws = {}
for i in range(len(headers)):
    zongo_aws[headers[i]] = dat[:, i]

# import flux file
dat = np.genfromtxt(zongo_file_flux, delimiter=',', skip_header=7)  # ignore first column as time
headers = np.genfromtxt(zongo_file_flux, delimiter=',', skip_header=6, skip_footer=17569, dtype=(str))
# put into dictionary
zongo_flux = {}
for i in range(len(headers)):
    zongo_flux[headers[i]] = dat[:, i]

zongo_alb = pd.read_csv(zongo_file_albedo, skiprows=2, index_col=0, parse_dates=[0], usecols=(0, 1), nrows=334)


times = pd.date_range(start='1999-09-01 00:00', end='2000-08-30 23:00', freq='1H',
                      tz=zongo.tz)  # AWS data has timestamp in day of year, and appears to end at end of 30th August, as was leap year.
times_flux = pd.date_range(start='1999-09-01 00:00', end='2000-09-01 00:00', freq='30min',
                           tz=zongo.tz)  # Flux data has d/m/yyyy timestamp, but has gaps - looks like the data has been shifted - see below

zongo_aws_pd = pd.DataFrame.from_dict(zongo_aws)
zongo_aws_pd.index = times

zongo_flux_pd = pd.DataFrame.from_dict(zongo_flux)
zongo_flux_pd.index = times_flux

# add albedo to AWS data - noting that
zongo_aws_pd['albedo'] = np.full(zongo_aws_pd.shape[0], np.nan)
zongo_aws_pd['albedo'][:8016] = blockfun(zongo_alb.albedo.values, -24)

# add flux  to AWS data in two parts, as dates don't line up and comparison of U * (t- ts) looks better if flux has a missing day around 7 feb 2000
zongo_aws_pd['H'] = np.full(zongo_aws_pd.shape[0], np.nan)
zongo_aws_pd['LE'] = np.full(zongo_aws_pd.shape[0], np.nan)
zongo_aws_pd['H'][0] = zongo_flux_pd['H (Wm-2)'][0]  # only one 30-min flux period covering first hour of AWS data
zongo_aws_pd['LE'][0] = zongo_flux_pd['LE (Wm-2)'][0]
h_fluxes = blockfun(zongo_flux_pd['H (Wm-2)'][1:],
                    2)  # average to hourly, missing the first 30-min period. timestamps all assumed to be the end of the time range
le_fluxes = blockfun(zongo_flux_pd['LE (Wm-2)'][1:], 2)
zongo_aws_pd['H'][1:3912] = h_fluxes[
                            :3911]  # averaged flux data starts 1 hour after AWS data. use data from stated timestamp till '2000-02-11 00:00:00-0400', tz='America/La_Paz', based on visual inspection of exchange coefficient based on h_fluxes/ h
zongo_aws_pd['LE'][1:3912] = le_fluxes[:3911]
zongo_aws_pd['H'][3912:] = h_fluxes[
                           3911 + 24:-1]  # flux is 25 hours longer than AWS file, but we are moving the second part of the flux data forward 1 day, so is reduced to 1 hour
zongo_aws_pd['LE'][3912:] = le_fluxes[
                            3911 + 24:-1]  # flux is 25 hours longer than AWS file, but we are moving the second part of the flux data forward 1 day, so is reduced to 1 hour

zongo_aws_pd.to_pickle(outfile_aws)

