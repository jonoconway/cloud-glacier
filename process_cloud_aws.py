"""
code to homogenise datasets, add calculated clear-sky SW and LW to AWS data, and calculate cloud metrics

    shortwave is calculated using the simplied Solis mode
    	Ineichen, P. (2008). A broadband simplified version of the Solis clear sky model. Solar Energy, 82(8), 758-762. doi:10.1016/j.solener.2008.02.009

	clear-sky code from pvlib library
	https://pvlib-python.readthedocs.io/en/stable/clearsky.html#simplified-solis

	longwave is calculated using Konzelmann 1994 scheme

Jono Conway
jono.conway@niwa.co.nz
"""
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pvlib.location import Location, solarposition
from calc_cloud_metrics import *

if __name__ == '__main__':

    sites = ['brewster', 'chhota_shigri', 'conrad_abl', 'conrad_acc', 'guanaco', 'kersten', 'langfjordjokelen','mera_summit',
              'midtdalsbreen', 'morteratsch', 'naulek', 'nordic', 'qasi', 'storbreen', 'yala', 'zongo']  #'toro',

    for site in sites:

        print('processing {}'.format(site))

        if site == 'brewster':
            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/brewster.pkl"
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/brewster_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = -44.08
            lon = 169.43
            alt = 1760
            name = 'Brewster Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2010, 10, 26, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2012, 9, 1, 00)].index, inplace=True)

            aod700 = 0.1
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'tc_flux': "tc", 'rh': "rh", 'pres': 'pres', 'v_flux': 'ws', 'swin_flux': 'swin', 'swout': 'swout', 'lwin_flux': 'lwin',
                         'lwout_flux': 'lwout',
                         'QS': 'qs', 'QL': 'ql', 'wd': 'wd', 'accalb_flux': 'alb', 'QM': 'qm', 'QC': 'qc', 'QPRC': 'qr', 'QPS': 'qps', 'meltmass': 'meltmass',
                         'submass': 'submass'}, inplace=True)
            aws_df['lwout'] = aws_df['lwout'] * -1  # change sign

        elif site == 'chhota_shigri':
            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/chhota_shigri.pkl"
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/chhota_shigri_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 32.28
            lon = 77.58
            alt = 4670
            name = 'Chhota Shigri Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2012, 10, 1, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2013, 9, 5, 00)].index, inplace=True)

            aod700 = 0.1
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'Ta high': "tc", 'RH high': "rh", 'WS high': 'ws', 'SWI from Acc method': 'swin', 'SWO': 'swout', 'LWI': 'lwin',
                         'LWO': 'lwout', 'H': 'qs', 'LE': 'ql', 'Albedo acc': 'alb', 'Melt (m we).1': 'meltmass', 'G': 'qc', 'Ssub': 'qps',
                         'Subli (cmwe).1': 'submass'}, inplace=True)
            aws_df['submass'] = aws_df['submass'] / 100  # convert from cmw.e. to m w.e.
            aws_df['qm'] = aws_df['meltmass'] * 3865.740741  # calculate melt energy from melt rate


        elif site == 'conrad_2015_st1':

            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2015_st1.pkl"
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2015_st1_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 50.82486
            lon = -116.92247
            alt = 2138
            tz = 'Etc/GMT+9'
            name = 'Conrad Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2015, 7, 18, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2015, 9, 6, 00)].index, inplace=True)

            aod700 = 0.05
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'T_2m': "tc", 'RH_2m': "rh", 'Pressure': 'pres', 'U_2m': 'ws', 'SW_down': 'swin', 'SW_up': 'swout', 'LW_down': 'lwin',
                         'LW_up': 'lwout',
                         'QH': 'qs', 'QE': 'ql', 'U_dir': 'wd', 'albedo': 'alb'}, inplace=True)


        elif site == 'conrad_2015_st2':

            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2015_st2.pkl"
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2015_st2_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 50.82306
            lon = -116.92128
            alt = 2163
            tz = 'Etc/GMT+9'
            name = 'Conrad Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2015, 7, 18, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2015, 9, 6, 00)].index, inplace=True)

            aod700 = 0.05
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'T_2m': "tc", 'RH_2m': "rh", 'Pressure': 'pres', 'U_2m': 'ws', 'SW_down': 'swin', 'SW_up': 'swout', 'LW_down': 'lwin',
                         'LW_up': 'lwout',
                         'QH': 'qs', 'QE': 'ql', 'U_dir': 'wd', 'albedo': 'alb'}, inplace=True)


        elif site == 'conrad_2016_st1':

            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2016_st1.pkl"
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2016_st1_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 50.82303
            lon = -116.91992
            alt = 2164
            tz = 'UTC'
            name = 'Conrad Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2016, 6, 19, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2016, 8, 27, 00)].index, inplace=True)

            aod700 = 0.05
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'T_2m': "tc", 'RH_2m': "rh", 'Pressure': 'pres', 'U_2m': 'ws', 'SW_down': 'swin', 'SW_up': 'swout', 'LW_down': 'lwin',
                         'LW_up': 'lwout',
                         'QH': 'qs', 'QE': 'ql', 'U_dir': 'wd', 'albedo': 'alb'}, inplace=True)


        elif site == 'conrad_acc':

            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2016_st2.pkl"
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_acc_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 50.78219
            lon = -116.91197
            alt = 2909
            tz = 'UTC'
            name = 'Conrad Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2016, 6, 17, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2016, 8, 24, 00)].index, inplace=True)

            aod700 = 0.05
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'T_2m': "tc", 'RH_2m': "rh", 'Pressure': 'pres', 'U_2m': 'ws', 'SW_down': 'swin', 'SW_up': 'swout', 'LW_down': 'lwin',
                         'LW_up': 'lwout',
                         'QH': 'qs', 'QE': 'ql', 'U_dir': 'wd', 'albedo': 'alb'}, inplace=True)

        elif site == 'conrad_abl':

            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2015_st2.pkl"
            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 50.82306
            lon = -116.92128
            alt = 2163
            tz = 'Etc/GMT+9'
            name = 'Conrad Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2015, 7, 18, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2015, 9, 6, 00)].index, inplace=True)
            aws_df_2015_st2 = aws_df.copy()

            infile = "C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_2016_st1.pkl"
            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 50.82303
            lon = -116.91992
            alt = 2164
            tz = 'UTC'
            name = 'Conrad Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2016, 6, 19, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2016, 8, 27, 00)].index, inplace=True)
            aws_df_2016_st1 = aws_df.copy()

            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/conrad_abl_withcloud.pkl'
            blank_datetimes = pd.date_range(aws_df_2015_st2.stime[-1] + dt.timedelta(hours=1),aws_df_2016_st1.stime[0] - dt.timedelta(hours=1),freq='1H')
            blank_df = pd.DataFrame(np.full(blank_datetimes.shape,np.nan),index=blank_datetimes)
            aws_df = pd.concat([aws_df_2015_st2, blank_df, aws_df_2016_st1])

            # rename variables
            aws_df.rename(
                columns={'T_2m': "tc", 'RH_2m': "rh", 'Pressure': 'pres', 'U_2m': 'ws', 'SW_down': 'swin', 'SW_up': 'swout', 'LW_down': 'lwin',
                         'LW_up': 'lwout',
                         'QH': 'qs', 'QE': 'ql', 'U_dir': 'wd', 'albedo': 'alb'}, inplace=True)

            aod700 = 0.05
            svf = 1

        elif site == 'guanaco':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/guanaco.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/guanaco_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = -29.34
            lon = -70.01
            alt = 5324
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name='Guanaco Glacier')
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2008, 11, 1, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2011, 4, 30, 00)].index, inplace=True)

            aod700 = 0.01
            svf = 1

            # rename variables
            aws_df.rename(columns={'tc': "tc", 'rh': "rh", 'pres': 'pres', 'ws': 'ws', 'swin': 'swin', 'lwin': 'lwin',
                                   'albedo': 'alb', 'QS': 'qs', 'QL': 'ql', 'QPRC': 'qr', 'QC': 'qc', 'QM': 'qm'}, inplace=True)
            aws_df['alb'] = blockfun(aws_df['alb'].values[::24], -24)  # move albedo to align with the day again.
            aws_df['swout'] = aws_df['swin'] * aws_df['alb']


        elif site == 'kersten':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/kersten.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/kersten_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60

            name = 'Kersten Glacier'
            lat = -3.078
            lon = 37.354
            alt = 5873
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2005, 2, 9, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2008, 1, 23, 00)].index, inplace=True)

            aod700 = 0.01
            svf = 1 # set to 1 as CNR1 instrument used with 150 and svf is 0.96

            # rename variables
            aws_df.rename(columns={'tc': "tc", 'rh': "rh", 'pres': 'pres', 'ws': 'ws', 'swin': 'swin', 'lwin': 'lwin', 'lwout': 'lwout_aws', 'LWout': 'lwout',
                                   'alpha': 'alb', 'QS': 'qs', 'QL': 'ql', 'QC': 'qc', 'QPS': 'qps', 'QM': 'qm'}, inplace=True)
            aws_df['lwout'] = aws_df['lwout'] * -1

        elif site == 'langfjordjokelen':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/langfjordjokelen.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/langfjordjokelen_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 70.133
            lon = 21.75
            alt = 650
            tz = 'UTC'
            name = 'Langfjordjokelen Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)  # in UTC
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2007, 9, 14, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2010, 8, 19, 00)].index, inplace=True)

            aod700 = 0.02
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'Tair': "tc", 'RH': "rh", 'pres': 'pres', 'FF': 'ws', 'SWin_corr': 'swin', 'SWout': 'swout', 'LWin': 'lwin', 'LWout_model': 'lwout',
                         'albrun': 'alb', 'Tsurf_calc': 'ts', 'Hsen': 'qs', 'Hlat': 'ql', 'Gs': 'qc', 'melt_energy': 'qm'}, inplace=True)
            # standardise units
            aws_df.pres = aws_df.pres / 100  # convert from Pa to hPa


        elif site == 'naulek':
            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/mera_naulek_combined.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/naulek_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60

            lat = 27.7177
            lon = 86.8974
            alt = 5380
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name='Naulek Glacier')
            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2013, 1, 1, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2017, 12, 31, 00)].index, inplace=True)

            aod700 = 0.01
            svf = 1  # used to correct LW data in clear-sky. if CNR1, view is only 150 degrees, so set to 1 if actual svf

            aws_df.rename(
                columns={'air_TC2': "tc", 'air_RH2': "rh", 'windspeed': 'ws', 'SW_up': 'swin', 'SW_dw': 'swout', 'LW_up_cor': 'lwin', 'LW_dw_cor': 'lwout',
                         'albedo': 'alb', 'Tsurf_calc': 'ts', 'Hbulk': 'qs', 'LEbulk': 'ql'}, inplace=True)

            mask = np.isnan(aws_df.tc.values)
            aws_df.loc[mask] = np.nan
            alb_day = blockfun(aws_df['swout'].values, 24, method='sum') / blockfun(aws_df['swin'].values, 24, method='sum')
            aws_df['alb'] = blockfun(alb_day, -24)

        elif site == 'mera_summit':
            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/mera_summit_combined.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/mera_summit_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60

            lat = 27.7065
            lon = 86.8737
            alt = 6342
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name='Mera Glacier summit')
            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2013, 11, 21, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2016, 7, 26, 00)].index, inplace=True)

            aod700 = 0.01
            svf = 1  # used to correct LW data in clear-sky. if CNR1, view is only 150 degrees, so set to 1 if actual svf is

            aws_df.rename(
                columns={'air_TC2': "tc", 'air_RH2': "rh", 'windspeed': 'ws', 'SW_up': 'swin', 'SW_dw': 'swout', 'LW_up_cor': 'lwin', 'LW_dw_cor': 'lwout',
                         'albedo': 'alb', 'Tsurf_calc': 'ts', 'Hbulk': 'qs', 'LEbulk': 'ql'}, inplace=True)

            mask = (aws_df.stime > dt.datetime(2015, 9, 16, 00)) & (
                    aws_df.stime <= dt.datetime(2015, 9, 19, 00))  # remove period that likely has rime on sw sensor
            aws_df.loc[mask] = np.nan
            alb_day = blockfun(aws_df['swout'].values, 24, method='sum') / blockfun(aws_df['swin'].values, 24, method='sum')
            aws_df['alb'] = blockfun(alb_day, -24)

        elif site == 'midtdalsbreen':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/midt_00-06.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/midtdalsbreen_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 60.5667
            lon = 7.4667
            alt = 1450
            tz = 'UTC'
            name = 'Midtdalsbreen Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)  # in UTC
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2000, 11, 1, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2006, 9, 8, 00)].index, inplace=True)

            aod700 = 0.02
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'Tair': "tc", 'RH': "rh", 'pres': 'pres', 'FF': 'ws', 'SWin_corr': 'swin', 'SWout': 'swout', 'LWin': 'lwin', 'LWout_model': 'lwout',
                         'albrun': 'alb', 'Tsurf_calc': 'ts', 'Hsen': 'qs', 'Hlat': 'ql', 'Gs': 'qc', 'melt_energy': 'qm'}, inplace=True)

            # standardise units
            aws_df.pres = aws_df.pres / 100  # convert from Pa to hPa


        elif site == 'morteratsch':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/morteratsch.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/morteratsch_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 46.4221
            lon = 9.9318
            alt = 2100
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name='Morteratsch')  # in UTC
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(1998, 7, 10, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2007, 5, 15, 00)].index, inplace=True)

            aod700 = 0.2
            svf = 1

            # rename variables, choosing which to use for cloud metrics. SWin_corr is corrected using albedo and swout
            aws_df.rename(
                columns={'Tair': "tc", 'RH': "rh", 'pres': 'pres', 'FF': 'ws', 'SWin_corr': 'swin', 'SWout': 'swout', 'LWin': 'lwin', 'LWout_model': 'lwout',
                         'albrun': 'alb', 'Tsurf_calc': 'ts', 'Hsen': 'qs', 'Hlat': 'ql', 'Gs': 'qc', 'melt_energy': 'qm'}, inplace=True)
            # standardise units
            aws_df.pres = aws_df.pres / 100  # convert from Pa to hPa


        elif site == 'nordic':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/nordic.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/nordic_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 53.05083
            lon = -120.4443
            alt = 2208
            tz = 'UTC'
            name = 'Nordic Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2014, 7, 12, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2014, 8, 27, 00)].index, inplace=True)

            aod700 = 0.05
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'T_2m': "tc", 'RH_2m': "rh", 'Pressure': 'pres', 'U_2m': 'ws', 'SW_down': 'swin', 'SW_up': 'swout', 'LW_down': 'lwin',
                         'LW_up': 'lwout', 'QH': 'qs', 'QE': 'ql', 'U_dir': 'wd', 'albedo': 'alb'}, inplace=True)


        elif site == 'qasi':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/qasi.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/qasi_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 64.1623
            lon = -51.3587
            alt = 710
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name='Qassigiannguit Glacier')
            # convert
            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2014, 7, 29, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2016, 10, 4, 00)].index, inplace=True)

            # remove times without valid SW data
            mask = (aws_df.stime > dt.datetime(2015, 3, 21, 00)) & (aws_df.stime <= dt.datetime(2015, 6, 24, 00))
            aws_df.loc[mask] = np.nan
            # aws_df.drop(aws_df[np.logical_and(aws_df.stime > dt.datetime(2015, 3, 21, 00), aws_df.stime <= dt.datetime(2015, 6, 24, 00))].index, inplace=True)

            aod700 = 0.01
            svf = 1

            # rename variables
            aws_df.rename(columns={'T_C': "tc", 'RH_%': "rh", 'P_hPa': 'pres', 'WS_ms-1': 'ws', 'SRin_Wm-2': 'swin', 'SRout_Wm-2': 'swout', 'LRin_Wm-2': 'lwin',
                                   'LRout_Wm-2': 'lwout', 'Rain_m': 'rain', 'Tsurf_C': 'ts', 'SHF_Wm-2': 'qs', 'LHF_Wm-2': 'ql',
                                   'GF_Wm-2': 'qc', 'rainHF_Wm-2': 'qr', 'MEsurf_Wm-2': 'qm', 'Hmelt_m': 'meltmass', 'Hsubl_m': 'submass'}, inplace=True)

            alb_day = blockfun(aws_df['swout'].values, 24, method='sum') / blockfun(aws_df['swin'].values, 24, method='sum')
            aws_df['alb'] = blockfun(alb_day, -24)  # Albedo column is hourly average

        elif site == 'storbreen':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/storbreen.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/storbreen_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
            lat = 61.583
            lon = 8.166
            alt = 1570
            tz = 'UTC'
            name = 'Storbreen Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)  # in UTC
            aws_df = process_timezones(aws_df, lon, datastep=datastep)
            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2001, 9, 7, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2006, 9, 8, 00)].index, inplace=True)

            aod700 = 0.02
            svf = 1

            # rename variables
            aws_df.rename(
                columns={'Tair': "tc", 'RH': "rh", 'pres': 'pres', 'FF': 'ws', 'SWin_corr': 'swin', 'SWout': 'swout', 'LWin': 'lwin', 'LWout_model': 'lwout',
                         'albrun': 'alb', 'Tsurf_calc': 'ts', 'Hsen': 'qs', 'Hlat': 'ql', 'Gs': 'qc', 'melt_energy': 'qm'}, inplace=True)
            # standardise units
            aws_df.pres = aws_df.pres / 100  # convert from Pa to hPa


        elif site == 'yala':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/yala_combined.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/yala_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60

            lat = 28.235  # coordinates from Kok 2019 IJC
            lon = 85.618
            alt = 5350
            name = 'Yala Glacier'
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name=name)

            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(2014, 5, 9, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2018, 4, 26, 00)].index, inplace=True)

            aod700 = 0.01
            svf = 1  # used to correct LW data in clear-sky. if CNR1, view is only 150 degrees, so set to 1 if actual svf is

            aws_df.rename(
                columns={'TAIR_aws': "tc", 'RH_aws': "rh", 'WSPD_aws': 'ws', 'KINC_aws': 'swin', 'KOUT_aws': 'swout', 'LINC_aws': 'lwin', 'LOUT_flux': 'lwout',
                         'Hbulk': 'qs', 'LEbulk': 'ql'},
                inplace=True)
            alb_day = blockfun(aws_df['swout'].values, 24, method='sum') / blockfun(aws_df['swin'].values, 24, method='sum')
            aws_df['alb'] = blockfun(alb_day, -24)
            mask = np.isnan(aws_df.tc.values)
            aws_df.loc[mask] = np.nan


        elif site == 'zongo':

            infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/zongo_aws_with_qm.pkl'
            outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/zongo_withcloud.pkl'

            aws_df = pickle.load(open(infile, 'rb'))
            datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60

            lat = -16.25
            lon = -68.167
            alt = 5040
            aws_loc = Location(lat, lon, tz=aws_df.index.tz, altitude=alt, name='Zongo Glacier')

            aws_df = process_timezones(aws_df, lon, datastep=datastep)

            aws_df.drop(aws_df[aws_df.stime <= dt.datetime(1999, 9, 2, 00)].index, inplace=True)
            aws_df.drop(aws_df[aws_df.stime > dt.datetime(2000, 8, 29, 00)].index, inplace=True)

            aod700 = 0.01
            svf = 1

            aws_df.rename(columns={'T': "tc", 'RH': "rh", 'U': 'ws', 'Swinc': 'swin', 'Lwatm': 'lwin',
                                   'Lwsurf': 'lwout', 'Prec3h': 'precip', 'albedo': 'alb', 'sensible': 'qs', 'latent': 'ql'}, inplace=True)

            aws_df['qm'] = aws_df['melt(mm)'] / 0.01078
            aws_df['swout'] = aws_df['swin'] * aws_df['alb']

            mask = (aws_df.stime > dt.datetime(2000, 1, 11, 10)) & (aws_df.stime <= dt.datetime(2000, 1, 11, 23))
            aws_df.loc[mask] = np.nan


        else:

            print('incorrect site chosen')

        # basic QA
        aws_df.loc[aws_df.swin < 0, 'swin'] = 0
        aws_df.loc[aws_df.swout < 0, 'swout'] = 0
        aws_df.loc[aws_df.lwout > 315.6, 'lwout'] = 315.6
        if 'ts' not in aws_df.keys():
            aws_df['ts'] = ((aws_df['lwout'].values / 5.67e-8) ** 0.25) - 273.15
            aws_df.loc[aws_df.ts > 0, 'ts'] = 0
        if 'qm' not in aws_df.keys():
            if 'qs' in aws_df.keys() and 'ql' in aws_df.keys() and 'lwout' in aws_df.keys() and 'alb' in aws_df.keys():
                seb = aws_df['swin'] * (1 - aws_df['alb']) + aws_df['lwin'] - aws_df['lwout'] + aws_df['qs'] + aws_df['ql']
                qm = seb
                qm[np.logical_or(aws_df['ts'] < -0.1, seb < 0)] = 0  # no melt when ts not near 0 or negative seb
                aws_df['qm'] = qm

        datastep = (aws_df.index.to_pydatetime()[1] - aws_df.index.to_pydatetime()[0]).total_seconds() / 60
        blocksize = int(1440 / datastep)

        # calculate surface met variables
        tc = aws_df['tc'].values
        tk = tc + 273.15
        aws_df['tk'] = tk
        rh = aws_df['rh'].values
        ea = ea_from_tc_rh(tc, rh)
        w = w_from_ea_tk(ea, tk)
        aws_df['w'] = w
        aws_df['ea'] = ea
        lwin = aws_df['lwin'].values
        swin = aws_df['swin'].values

        #optimise clear-sky longwave
        m = 7  # set coefficent back to 7 according to DURR 2004
        ind_nonan = np.logical_and(~np.isnan(aws_df.lwin.values), ~np.isnan(aws_df.ea.values), ~np.isnan(aws_df.tk.values))
        ind_nonan_rh80 = np.logical_and(ind_nonan, (aws_df.rh.values < 80))
        b, b_rmsd, ead = optimise_lw_cs(aws_df.lwin.values[ind_nonan_rh80], aws_df.ea.values[ind_nonan_rh80], aws_df.tk.values[ind_nonan_rh80], svf=svf,
                                        alt=aws_loc.altitude, m=m, return_stats=True)
        ind10 = find_cs_from_lw(aws_df.lwin.values[ind_nonan_rh80], aws_df.ea.values[ind_nonan_rh80], aws_df.tk.values[ind_nonan_rh80])

        ### store configuration
        config = {}
        config['blocksize'] = blocksize
        config['b'] = b
        config['b_rsmd'] = np.round(b_rmsd, 4)
        config['aod700'] = aod700
        config['svf'] = svf
        config['m'] = m

        # calc clear-sky shortwave
        aws_df = add_clear_sky(aws_df, aws_loc, method='pvlib')
        sw_pot = aws_df.sw_cs_ghi.values
        times2 = pd.DatetimeIndex(aws_df.utc_time) - dt.timedelta(minutes=datastep / 2)
        solpos = solarposition.get_solarposition(times2, aws_loc.latitude, aws_loc.longitude)
        apparent_zenith = solpos['apparent_zenith'].values
        apparent_elevation = solpos['apparent_elevation'].values
        # set all values where sun is below horizon to 0
        aws_df.loc[apparent_zenith > 90, 'swin'] = aws_df.loc[apparent_zenith > 90, 'swin'].values * 0
        swin[apparent_zenith > 90] = swin[apparent_zenith > 90] * 0
        # define filter for daytime/ illuminated periods
        not_daytime = np.zeros(tk.shape, dtype=bool)
        not_daytime[apparent_zenith > 80] = True  # removes values at low zenith angles that may not be illuminated
        not_daytime[swin == 0] = True  # removes values when there are no measurements or at night

        # shortwave metrics
        trc = swin / sw_pot  # basic clear-sky transmissivity for each datapoint
        trc[trc > 2] = 2  # values > 2 likely incorrect
        neff = neff_from_trc(trc, vp=None)  # hourly values
        neff_sum24 = neff_sum24_from_sw(swin, sw_pot, blocksize, not_daytime, vp=ea)  # hourly values based on daily sums
        d_swinsum = blockfun(swin, blocksize, not_daytime, method='sum', keep_nan=True)
        d_swpotsum = blockfun(sw_pot, blocksize, not_daytime, method='sum', keep_nan=True)
        d_trc = dtrc_from_sw(swin, sw_pot, blocksize, not_daytime, keep_nan=True)  # daily average based on daily sums
        d_neff_sum24 = blockfun(neff_sum24, blocksize, not_daytime, keep_nan=True)  # daily average based on daily sums

        # longwave metrics
        ecs = ecs_from_ea_ta(ea, tc, ead=ead, b=b, m=m)  # clear-sky emissivity
        lw_cs = ecs * sigma * tk ** 4  # clear-sky longwave
        eeff = eeff_from_lw_tk(lwin, tk, svf)  # effective emissivitiy of atmosphere
        nep = nep_from_ecs(eeff, ecs)  # longwave equivalent cloudiness
        d_nep = blockfun(nep, blocksize, keep_nan=True)  # average over 24 hours
        d_nep_day = blockfun(nep, blocksize, filter=not_daytime, keep_nan=True)  # average of daytime illuminated values
        # create 30-min timeseries of daily average nep
        nep_av24 = blockfun(d_nep, -1 * blocksize)
        nep_av24_day = blockfun(d_nep_day, -1 * blocksize)

        #store hourly data
        aws_df['trc'] = trc
        aws_df['neff'] = neff
        aws_df['neff_sum24'] = neff_sum24
        aws_df['ecs'] = ecs
        aws_df['lw_cs'] = lw_cs
        aws_df['eeff'] = eeff
        aws_df['nep'] = nep
        aws_df['nep_av24'] = nep_av24
        aws_df['nep_av24_day'] = nep_av24_day
        aws_df['apparent_zenith'] = apparent_zenith
        # create net radiation components
        aws_df['swnet'] = aws_df['swin'] * (1 - aws_df['alb'])
        aws_df['lwnet'] = aws_df['lwin'] - aws_df['lwout']
        aws_df['rnet'] = aws_df['swnet'] + aws_df['lwnet']

        # create output for daily data
        # daily_df = aws_df.resample('1D', label='left', closed='right').mean() # not using as excludes nan values in creating daily values
        # daily_df.index = daily_df.index + dt.timedelta(hours=12) # move to the middle of day before
        # puts the timestamp at 12:00 hours on the day of the average. ie. average from '2014-07-29 01:00:00' to '2014-07-30 00:00:00' is put at '2014-07-29 12:00:00'
        d = {}
        for var in aws_df.keys():
            try:
                d[var] = blockfun(aws_df[var].values, 24, keep_nan=True)
            except:
                print('skipping {}'.format(var))

        daily_df = pd.DataFrame.from_dict(d)
        daily_ind = pd.date_range(start=aws_df.index[0].date(), end=aws_df.index[-1].date(), freq='1D')
        daily_df.index = daily_ind + dt.timedelta(hours=12)
        daily_df.index = daily_df.index.set_names('stime')

        daily_df['d_trc'] = d_trc
        daily_df['d_neff_sum24'] = d_neff_sum24
        daily_df['d_nep'] = d_nep
        daily_df['d_nep_day'] = d_nep_day
        daily_df['d_swinsum'] = d_swinsum
        daily_df['d_swpotsum'] = d_swpotsum

        # number of nan values in each column
        daily_df.isnull().values.sum(axis=0)

        save_dict = {}
        save_dict['aws_df'] = aws_df
        save_dict['config'] = config
        save_dict['daily_df'] = daily_df
        save_dict['aws_loc'] = aws_loc

        pickle.dump(save_dict, open(outfile, 'wb'), -1)


