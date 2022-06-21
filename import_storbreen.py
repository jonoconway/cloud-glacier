"""
code to read in data from AWS
"""

import pandas as pd
import datetime as dt
from pvlib.location import Location

aws_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Norway/Stor_01-06_fluxes_toJono.csv"
outfile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/storbreen.pkl"

# import AWS file
aws_df = pd.read_csv(aws_file, header=0)
lat = 61.583
lon = 8.166
alt = 1570
tz = 'UTC'
name = 'Storbreen Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt,name=name)

aws_times = pd.date_range(start='2001-9-6 11:30', end='2006-9-8 11:00', freq='30min', tz=aws_loc.tz)# timestamp is given in the middle of the averaging period, but move to standard at end of averaging period

aws_df.index = aws_times

aws_df.to_pickle(outfile)


