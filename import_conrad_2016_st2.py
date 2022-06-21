"""
code to read in data from AWS
"""

import pandas as pd
from pvlib.location import Location

aws_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Nordic-Castle-Conrad/Met data for glaciers/Conrad_met_observations_hourly_st2_2016.csv"
outfile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2016_st2.pkl"

# import AWS file
aws_df = pd.read_csv(aws_file, header=0)
lat = 50.78219
lon = -116.91197
alt = 2909
tz = 'UTC'
name = 'Conrad Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt,name=name)

aws_times = pd.date_range(start='2016-6-17 01:00', end='2016-8-25 00:00', freq='1H', tz=aws_loc.tz)

aws_df.index = aws_times

aws_df.to_pickle(outfile)

