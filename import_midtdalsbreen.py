"""
code to read in data from AWS
"""

import pandas as pd
import datetime as dt
from pvlib.location import Location

aws_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Norway/Midt_00-06_fluxes_toJono.csv"
outfile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/midt_00-06.pkl"

# import AWS file
aws_df = pd.read_csv(aws_file, header=0)
lat = 60.5667
lon = 7.4667
alt = 1450
tz = 'UTC'
name = 'Midtdalsbreen Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt,name=name)

aws_times = pd.date_range(start='2000-10-1 10:00', end='2006-9-8 11:00', freq='30min', tz=aws_loc.tz)# timestamp is given in the middle of the averaging period, but move to standard at end of averaging period

aws_df.index = aws_times

aws_df.to_pickle(outfile)


