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

file_flux = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Kili/SEBcell32.dat"
file_AWS = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Kili/aws3kili-3years.dat"
file_mb = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Kili/MBcell32.dat"
outfile_aws = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/kersten.pkl'

# import AWS file

seb_headers = ['GR', 'swiasky', 'alpha', 'LWin', 'LWout', 'TS', 'QS', 'QL', 'QC', 'QPS', 'QM']
aws_headers = ['year', 'month', 'doy', 'dechour', 'tc', 'zh', 'rh', 'ws', 'zm', 'pres', 'solid_precip', 'swin', 'swout', 'lwin', 'lwout', 'ea']
mb_names = ['sfacc','depmass','suimass','refsmass','meltmass','submass']
aws_pd = pd.read_csv(file_AWS, delimiter='\t', names=aws_headers)
flux_pd = pd.read_csv(file_flux, delimiter='\t', names=seb_headers)
mb_pd = pd.read_csv(file_mb,delimiter='\t',names=mb_names)

# estimated slope: 18

lat = -3.078
lon = 37.354
alt = 5873
tz_offset = 3  # hours offset from UTC of timestamp
if tz_offset == 0:
    tz = 'UTC'
else:
    tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
name = 'Kersten Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)

times = pd.date_range(start='2005-02-09 01:00', end='2008-01-24 00:00', freq='1H', tz=aws_loc.tz)

aws_pd.index = times
flux_pd.index = times
mb_pd.index = times

full_pd = pd.merge(aws_pd, flux_pd, how='outer', left_index=True, right_index=True, suffixes=['_aws', '_flux'])
full_pd = pd.merge(full_pd, mb_pd, how='outer', left_index=True, right_index=True)
full_pd.to_pickle(outfile_aws)


