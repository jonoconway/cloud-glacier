# cloud-glacier #
Analysis and visualisation of glacier surface energy balance datasets with focus on clouds and melt.

Author: Jono Conway - *jono.conway@niwa.co.nz*

Code corresponds to the paper "Cloud forcing of surface energy balance from in-situ measurements in diverse mountain glacier environments" by Conway et al.
https://tc.copernicus.org/preprints/tc-2022-24/

The code can be used to calculate cloud metrics for any AWS if the required data variables are available. The datasets used in the paper need to be obtained from authors listed in the paper.

The pvlib package is used for solar radiation calculations, while longwave and cloudiness calculations comes from Conway et al. 2015 https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.4014 and references therein.

Order scripts were run for paper:
- A. Import files from each site individually using various scripts 'import_##site##.py', where ##site## shoud be replaced with the name of the dataset
- B. Run process_cloud_aws.py to homogenise data, adding calculated clear-sky radiation and cloud metrics each dataset 
- C. Run monthly_processing.py to collate dataset and calculate monthly average cloud metrics. here breaks between clear-sky,partial cloud and overcast conditions are set
- D. Plot data using plot_cloud_bin.py and plot_monthly.py


### Required variables ###
- 'tc', air temperature (C)
- 'rh', relative humidity (%)
- 'ws', wind speed (m s^-1)
- 'swin', incoming shortwave radiation (W m^-2)
- 'swout' outgoing shortwave radiation (W m^-2) or albedo (0-1)
- 'lwin', incoming longwave radiation (W m^-2)
- 'lwout',incoming longwave radiation (W m^-2) or surface temperature (C)
- 'qs', turbulent sensible heat flux (W m^-2)
- 'ql', turbulent latent heat flux (W m^-2)

### Optional variables ###
- 'pres', air pressure (hPa)
- 'qc', heat from conduction into glacier surface (W m^-2)
- 'melt' melt rate (mm w.e.) or 'qm' melt energy (W m^-2)

### Required packages:
numpy
scipy
pandas
matplotlib
cartopy
pvlib
adjustText


### importing new datasets

1. set up a new file 'import_newsite.py'
this should read the data into a pandas dataframe with unique headers
specify the location attributes using
Location(latitude in degrees N, longitude in degrees E, timezone of data in compatiable format, altitude in metres, arbitrary name)
e.g. mort = Location(46.4167, 9.4167, tz='UTC', altitude=2100, name='Morteratsch')  # in UTC
specify the length of the record using
start and end times in timezone of dataset, timestep of data
e.g. times_mort = pd.date_range(start='1998-07-09 00:30', end='2007-05-15 00:00', freq='30min', tz=mort.tz)
2. Add an extra entry to process_cloud_aws.py
this is where you choose which variables correspond to the standard variables used in the processing, and where to cut the start and end of the record.