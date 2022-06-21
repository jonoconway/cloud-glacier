"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib

file_flux = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/fluxes/yalafluxes_trim.txt"
aws_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/processed_export_from_R/yalaGL_export.csv"
aws2014_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/processed_export_from_R/yalaGL2014_export.csv"
outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/yala_combined.pkl'

lat = 28.235  # coordinates from Kok 2019 IJC
lon = 85.618
alt = 5350
tz = 'Asia/Kathmandu'
# tz_offset = 6 # hours offset from UTC of timestamp
# if tz_offset == 0:
#     tz = 'UTC'
# else:
#     tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
name = 'Yala Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)

aws_df = pd.read_csv(aws_file, header=0)
aws_times = pd.date_range(start='2016-5-7 13:00', end='2018-4-26 08:00', freq='1H', tz=aws_loc.tz)
aws_df.index = aws_times

flux_df = pd.read_csv(file_flux, header=0)
times_flux = pd.date_range(start='2014-5-8 02:00', end='2017-12-31 01:00', freq='1H', tz=aws_loc.tz)
flux_df.index = times_flux

aws2014_df = pd.read_csv(aws2014_file, header=0)
times_aws2014 = pd.date_range(start='2014-5-8 02:00', end='2014-10-20 01:00', freq='1H',
                              tz=aws_loc.tz)  # shift one hour later to align with the flux file that is already shifted by dates
aws2014_df.index = times_aws2014
# rename 2014 aws columns to match 20
aws2014_df.rename(columns={'air_TC': 'TAIR', 'air_RH': 'RH', 'windspeed': 'WSPD', 'winddir': 'WINDDIR', 'SW_up': 'KINC', 'SW_dw': 'KOUT', 'LW_up_cor': 'LINC',
                           'LW_dw_cor': 'LOUT'}, inplace=True)
aws_df_full = pd.concat([aws2014_df, aws_df])

full_df = pd.merge(aws_df_full, flux_df, how='outer', left_index=True, right_index=True, suffixes=['_aws', '_flux'])

plt.plot(full_df.TAIR_flux - full_df.TAIR_aws)

# shifts should be the same as AWS file as starts before the flux file
shifts = [0,  16534, 25366]#7822,

for shift in shifts:
    if shift == 0:
        full_df[:-1] = full_df[1:].values.copy()
    else:
        full_df[shift:] = full_df[shift - 1:-1].values.copy()

full_df.to_pickle(outfile)
