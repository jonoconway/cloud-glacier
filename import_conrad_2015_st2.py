"""
code to read in data from AWS
"""

import pandas as pd
from pvlib.location import Location

aws_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Nordic-Castle-Conrad/Met data for glaciers/Conrad_met_observations_hourly_st2_2015.csv"
outfile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2015_st2.pkl"

# import AWS file
aws_df = pd.read_csv(aws_file, header=0)
lat = 50.82306
lon = -116.92128
alt = 2163
tz = 'Etc/GMT+9'
name = 'Conrad Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt,name=name)

aws_times = pd.date_range(start='2015-7-17 01:00', end='2015-9-6 00:00', freq='1H', tz=aws_loc.tz)

aws_df.index = aws_times

aws_df.to_pickle(outfile)
