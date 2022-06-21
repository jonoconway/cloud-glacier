"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib

summit_file_flux = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/fluxes/merasummitfluxes_trim.txt"
outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/mera_summit_combined.pkl'

# set up location
lat = 27.707  # coordinates from Kok 2019 IJC
lon = 86.874
alt = 6352
# tz_offset = 6  # hours offset from UTC of timestamp
# if tz_offset == 0:
#     tz = 'UTC'
# else:
#     tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
tz = 'Asia/Kathmandu' # 5.75 hour offset
name = 'Mera Glacier summit'
# import dateutil.tz as dttz # this method didn't work...
# tz = dttz.tzoffset('summit', 6.5*60)
# tz.zone = 'summit'
aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)

times_summit_flux = pd.date_range(start='2014-1-1 02:00', end='2016-8-2 20:00', freq='1H', tz=aws_loc.tz) # move forward by 30 mins.
# times_toro = times_toro = pd.date_range(start='2008-10-10 01:00', end='2011-12-21 00:00', freq='1H', tz=toro.tz)

summit_flux = pd.read_csv(summit_file_flux, header=0)
# summit_aws.drop(np.arange(43416,43426))
summit_flux.index = times_summit_flux

summit_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/processed_export_from_R/meraSU_export.csv"
times_summit = pd.date_range(start='2013-11-20 15:00', end='2016-8-2 19:00', freq='1H', tz=aws_loc.tz)
summit_aws = pd.read_csv(summit_file, header=0)
summit_aws.index = times_summit

# offsets found using import_mera_summit_Glacioclim.py. found midnight preceeding first change in SW data, and in 2016 moved to end of march,
# print(
# summit_aws.index[3105],
# summit_aws.index[3105+365*24],
# summit_aws.index[3105+365*2*24-48],
# summit_aws.index[3105+365*3*24+24],
# summit_aws.index[3105+365*4*24+24])

# join
full_pd = pd.merge(summit_aws, summit_flux, how='outer', left_index=True, right_index=True, suffixes=['_aws', '_flux'])

# # shifts should be the same as AWS file as starts before the flux file
shifts = [3105,11865,20577]

for shift in shifts:
    full_pd[shift:] = full_pd[shift - 1:-1].values.copy()

full_pd.to_pickle(outfile)

