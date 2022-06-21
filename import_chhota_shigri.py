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

file_summer = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Chhota Shigri Glacier/Summer 60 days_30 mins_aws.csv"
file_winter = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Chhota Shigri Glacier/Winter 60 days_30 mins_aws.csv"
file_postmon = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Chhota Shigri Glacier/Post monsoon 60 days_30 mins_aws.csv"
file_summer_seb = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Chhota Shigri Glacier/Summer 60 days_30 mins_seb.csv"
file_winter_seb = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Chhota Shigri Glacier/Winter 60 days_30 mins_seb.csv"
file_postmon_seb = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Chhota Shigri Glacier/Post monsoon 60 days_30 mins_seb.csv"
outfile_aws = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/chhota_shigri.pkl'

# import AWS file

winter_pd = pd.read_csv(file_winter,delimiter=',',header=0)
summer_pd = pd.read_csv(file_summer,delimiter=',',header=0)
postmon_pd = pd.read_csv(file_postmon,delimiter=',',header=0)

winter_pd_seb = pd.read_csv(file_winter_seb,delimiter=',',header=0)
summer_pd_seb = pd.read_csv(file_summer_seb,delimiter=',',header=0)
postmon_pd_seb = pd.read_csv(file_postmon_seb,delimiter=',',header=0)



lat = 32.28
lon = 77.58
alt = 4670
tz_offset = 6  # hours offset from UTC of timestamp
if tz_offset == 0:
    tz = 'UTC'
else:
    tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
name = 'Chhota Shigri Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)

postmon_times = pd.date_range(start='2012-10-01 00:00', end='2012-11-29 23:30', freq='30T', tz=aws_loc.tz)
winter_times = pd.date_range(start='2012-12-01 00:00', end='2013-01-29 23:30', freq='30T', tz=aws_loc.tz)
summer_times = pd.date_range(start='2013-07-08 00:00', end='2013-09-05 23:30', freq='30T', tz=aws_loc.tz)


# estimated slope: 18
postmon_full_pd = pd.merge(postmon_pd, postmon_pd_seb, how='outer', left_index=True, right_index=True, suffixes=['_aws', '_seb'])
winter_full_pd = pd.merge(winter_pd, winter_pd_seb, how='outer', left_index=True, right_index=True, suffixes=['_aws', '_seb'])
summer_full_pd = pd.merge(summer_pd, summer_pd_seb, how='outer', left_index=True, right_index=True, suffixes=['_aws', '_seb'])

postmon_full_pd.index = postmon_times
winter_full_pd.index = winter_times
summer_full_pd.index = summer_times

frames = [postmon_full_pd, winter_full_pd, summer_full_pd]
full_pd = pd.concat(frames)
full_pd.to_pickle(outfile_aws)

