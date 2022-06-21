"""
code to read in data from AWS

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib

qasi_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Qasigiannguit/659_SEB.csv"
outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/qasi.pkl'
# import AWS file

qasi = Location(64.1623, -51.3587, tz='Etc/GMT+1', altitude=710, name='Qassigiannguit Glacier')  # set to chile standard time

times_qasi = pd.date_range(start='2014-07-28 14:00', end='2016-10-04 13:00', freq='1H', tz=qasi.tz)
# times_toro = times_toro = pd.date_range(start='2008-10-10 01:00', end='2011-12-21 00:00', freq='1H', tz=toro.tz)

qasi_aws = pd.read_csv(qasi_file, header=0)
# qasi_aws.drop(np.arange(43416,43426))
qasi_aws.index = times_qasi

qasi_aws.to_pickle(outfile)

print()

