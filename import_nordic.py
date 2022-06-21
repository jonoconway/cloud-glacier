"""
code to read in data from AWS
"""

import pandas as pd
from pvlib.location import Location

aws_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Nordic-Castle-Conrad/Met data for glaciers/Nordic_met_observations_hourly_2014.csv"
outfile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/nordic.pkl"

# import AWS file
aws_df = pd.read_csv(aws_file, header=0)
lat = 53.05083
lon = -120.4443
alt = 2208
tz = 'UTC'
name = 'Nordic Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt,name=name)

aws_times = pd.date_range(start='2014-7-12 01:00', end='2014-8-28 00:00', freq='1H', tz=aws_loc.tz)

aws_df.index = aws_times

aws_df.to_pickle(outfile)


