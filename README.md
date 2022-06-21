# cloud-glacier
Analysis and visualisation of glacier surface energy balance datasets with focus on clouds and melt.
Author: Jono Conway
jono.conway@niwa.co.nz

Code corresponds to the paper "Cloud forcing of surface energy balance from in-situ measurements in diverse mountain glacier environments" by Conway et al.
https://tc.copernicus.org/preprints/tc-2022-24/

The code can be used to calculate cloud metrics for any AWS if the required data variables are available. The datasets used in the paper need to be obtained from authors listed in the paper.

The pvlib package is used for solar radiation calculations, while longwave and cloudiness calculations comes from Conway et al. 2015 https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.4014 and references therein.

Order scripts were run for paper:
A. import files from each site individually using various scripts 'import_##site##.py', where ##site## shoud be replaced with the name of the dataset
B. run process_cloud_aws.py to homogenise data, adding calculated clear-sky radiation and cloud metrics each dataset 
C. run monthly_processing.py to collate dataset and calculate monthly average cloud metrics. here breaks between clear-sky,partial cloud and overcast conditions are set
D. plot data using plot_cloud_bin.py and plot_monthly.py


Required variables
'tc', 
'rh',  
'ws', 
'swin',
'swout' or albedo
'lwin',
'lwout', or surface temperature
'qs', 
'ql',

optional variables
'pres', 
'qc',
'melt' or 'qm'

Required packages:
numpy
scipy
pandas
matplotlib
cartopy
pvlib
adjustText