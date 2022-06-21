"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import dateutil
from pvlib.location import Location
import pvlib
from calc_cloud_metrics import blockfun

file_flux = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Brewster/modelOUT_br1_output_fixed_B.csv"
file_AWS = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Brewster/BrewsterGlacier_Oct10_Sep12_prep3.csv"
outfile_aws = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/brewster.pkl'

# import AWS file
aws_pd = pd.read_csv(file_AWS, delimiter=',', header=0)
flux_pd = pd.read_csv(file_flux, delimiter=',', header=0)

lat = -44.08
lon = 169.43
alt = 1760
tz_offset = 12 # hours offset from UTC of timestamp
if tz_offset == 0:
    tz = 'UTC'
else:
    tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
name = 'Brewster Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)

times = pd.date_range(start='2010-10-25 00:30', end='2012-09-2 00:00', freq='30T', tz=aws_loc.tz)

aws_pd.index = times
flux_pd.index = times

full_pd = pd.merge(aws_pd, flux_pd,how='outer',left_index=True,right_index=True,suffixes=['_aws','_flux'])
full_pd.to_pickle(outfile_aws)

plt.show()
