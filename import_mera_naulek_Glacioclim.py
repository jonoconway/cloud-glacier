"""
code to read in data from AWS
data from Glacioclim used as check for data from Litt 2019.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from pvlib.location import Location
import pvlib

naulek_file = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Mera/processed_export_from_R/naulek_export_trim.csv"
outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/mera_naulek.pkl'
# import AWS file


lat = 27.718  # coordinates from Kok 2019 IJC
lon = 86.897
alt = 5380
tz_offset = 6 # hours offset from UTC of timestamp
if tz_offset == 0:
    tz = 'UTC'
else:
    tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
name = 'Naulek Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)


times_naulek = pd.date_range(start='2012-11-28 01:00', end='2017-11-11 00:00', freq='1H', tz=aws_loc.tz)
# times_toro = times_toro = pd.date_range(start='2008-10-10 01:00', end='2011-12-21 00:00', freq='1H', tz=toro.tz)

naulek_aws = pd.read_csv(naulek_file, header=0)
# naulek_aws.drop(np.arange(43416,43426))
naulek_aws.index = times_naulek



naulek_file_GC = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/Glacioclim Himalaya/naulekfinal2012_16.csv"
outfile_GC = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/mera_naulekGC.pkl'
# import AWS file


lat = 27.718  # coordinates from Kok 2019 IJC
lon = 86.897
alt = 5380
tz_offset = 6 # hours offset from UTC of timestamp
if tz_offset == 0:
    tz = 'UTC'
else:
    tz = 'Etc/GMT{}'.format(int(tz_offset * -1))
name = 'Naulek Glacier'

aws_loc = Location(lat, lon, tz=tz, altitude=alt, name=name)


times_naulek_GC = pd.date_range(start='2012-11-27 12:30', end='2016-11-21 10:30', freq='30T', tz=aws_loc.tz)
# times_toro = times_toro = pd.date_range(start='2008-10-10 01:00', end='2011-12-21 00:00', freq='1H', tz=toro.tz)

naulek_aws_GC = pd.read_csv(naulek_file_GC, header=0,skiprows=4)


# take out common timeframes
ind_nau = naulek_aws.index[:34907]
sw_nau = naulek_aws.SW_up.values.copy()[:34907]

naulek_aws_GC_hourly = naulek_aws_GC.resample('1H', label='left', closed='right', loffset=dt.timedelta(hours=1)).mean()
ind_gc = naulek_aws_GC_hourly.index[12:]
sw_gc = naulek_aws_GC_hourly.SWin.values[12:]

assert np.all(ind_gc == ind_nau)

plt.plot(ind_gc.to_pydatetime(),sw_nau-sw_gc)
np.where((np.abs(sw_nau-sw_gc))>1)[0][0]
sw_nau[2957:] = sw_nau[2956:-1]
plt.plot(ind_gc.to_pydatetime(),sw_nau-sw_gc)
np.where((np.abs(sw_nau-sw_gc))>1)[0][0]
sw_nau[11725:] = sw_nau[11724:-1]
plt.plot(ind_gc.to_pydatetime(),sw_nau-sw_gc)
np.where((np.abs(sw_nau-sw_gc))>1)[0][0]
sw_nau[20429:] = sw_nau[20428:-1]
plt.plot(ind_gc.to_pydatetime(),sw_nau-sw_gc)
np.where((np.abs(sw_nau-sw_gc))>1)[0][0]
sw_nau[29721:] = sw_nau[29720:-1]
plt.plot(ind_gc.to_pydatetime(),sw_nau-sw_gc)

print(
ind_nau[2957],
ind_nau[11725],
ind_nau[20429],
ind_nau[29721])

print(
naulek_aws.index[2951],
naulek_aws.index[2951+365*24],
naulek_aws.index[2951+365*2*24-48],
naulek_aws.index[2951+365*3*24+24],
naulek_aws.index[2951+365*4*24+24])


naulek_aws.to_pickle(outfile)

