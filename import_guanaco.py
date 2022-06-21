"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib

guanaco_aws_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Guanaco/Guanaco_hr_2008_10_final14_header.csv"
guanaco_flux_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Guanaco/guanaco_standard_run_fluxes.csv"
outfile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/guanaco.pkl"

# import AWS file
guanaco_aws = pd.read_csv(guanaco_aws_file, header=0)
guanaco_flux = pd.read_csv(guanaco_flux_file, header=0)

guanaco_combined = pd.concat([guanaco_aws, guanaco_flux], axis=1)

guanaco = Location(-29.34, -70.01, tz='Etc/GMT+4', altitude=5324, name='Guanaco Glacier')

times_guanaco = pd.date_range(start='2008-11-1 00:00', end='2011-4-30 23:00', freq='1H', tz=guanaco.tz)


guanaco_combined.index = times_guanaco

guanaco_combined.to_pickle(outfile)


