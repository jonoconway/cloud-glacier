"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib

naulek_file_flux = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/fluxes/naulekfluxes.txt"
outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/mera_naulek_combined.pkl'

# set up location
lat = 27.718  # coordinates from Kok 2019 IJC
lon = 86.897
alt = 5380
tz = 'Asia/Kathmandu'
# tz_offset = 6  # hours offset from UTC of timestamp
# if tz_offset == 0:
#     tz = 'UTC'
# else:
#     tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
name = 'Naulek Glacier'
# import dateutil.tz as dttz # this method didn't work...
# tz = dttz.tzoffset('naulek', 6.5*60)
# tz.zone = 'naulek'
aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)

times_naulek_flux = pd.date_range(start='2013-1-1 00:00', end='2017-12-31 00:00', freq='1H', tz=aws_loc.tz)
# times_toro = times_toro = pd.date_range(start='2008-10-10 01:00', end='2011-12-21 00:00', freq='1H', tz=toro.tz)

naulek_flux = pd.read_csv(naulek_file_flux, header=0)
# naulek_aws.drop(np.arange(43416,43426))
naulek_flux.index = times_naulek_flux

naulek_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/processed_export_from_R/naulek_export_trim.csv"
times_naulek = pd.date_range(start='2012-11-28 01:00', end='2017-11-11 00:00', freq='1H', tz=aws_loc.tz)
naulek_aws = pd.read_csv(naulek_file, header=0)
naulek_aws.index = times_naulek

# offsets found using import_mera_naulek_Glacioclim.py. found midnight preceeding first change in SW data, and in 2016 moved to end of march,
# print(
# naulek_aws.index[2951],
# naulek_aws.index[2951+365*24],
# naulek_aws.index[2951+365*2*24-48],
# naulek_aws.index[2951+365*3*24+24],
# naulek_aws.index[2951+365*4*24+24])

# join
full_pd = pd.merge(naulek_aws, naulek_flux, how='outer', left_index=True, right_index=True, suffixes=['_aws', '_flux'])

# shifts should be the same as AWS file as starts before the flux file
shifts = [2951, 11711, 20471, 29255, 38015]

for shift in shifts:
    full_pd[shift:] = full_pd[shift - 1:-1].values.copy()

full_pd.to_pickle(outfile)

