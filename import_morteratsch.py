"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib

infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Morteratsch/Mort_98-07_fluxes_toJono_trim.csv"
outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/morteratsch.pkl'
# import AWS file

mort = Location(46.4167, 9.4167, tz='UTC', altitude=2100, name='Morteratsch')  # in UTC

times_mort = pd.date_range(start='1998-07-09 00:30', end='2007-05-15 00:00', freq='30min', tz=mort.tz) # timestamp is given in the middle of the averaging period, but move to standard at end of averaging period
#times_mort = pd.date_range(start='1998-07-09 00:15', end='2007-05-14 23:45', freq='30min', tz=mort.tz)

mort_aws = pd.read_csv(infile, header=0)
mort_aws.index = times_mort

mort_aws.to_pickle(outfile)

