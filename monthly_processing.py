"""
plotting relationships between daily cloud metrics for different site

Jono Conway
jono.conway@niwa.co.nz
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pandas as pd


data_dir = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/'
plot_dir = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/plots/'
outfile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/collated_monthly_{}_{}_{}.pkl'

# set breakpoints in nep to use for 'clear-sky' and 'overcast' conditions
cs_thres = 0.2
ov_thres = 0.8
n_days_thres = 10

files = os.listdir(data_dir)
sites_to_plot = ['brewster', 'chhota_shigri',  'guanaco', 'kersten', 'langfjordjokelen',  'mera_summit', 'midtdalsbreen',
                 'morteratsch', 'naulek', 'qasi', 'storbreen', 'yala', 'zongo','conrad_abl','conrad_acc','nordic']
collated_dict = {}
for f in files:
    if 'withcloud' in f:
        full_dict = pickle.load(open(data_dir + '/' + f, 'rb'))
        s = '_'
        site = '{}'.format(s.join(f.split('_')[:-1]))
        if site in sites_to_plot:
            collated_dict[site] = full_dict


for site in collated_dict.keys():
    full_dict = collated_dict[site]
    daily_df = full_dict['daily_df']
    d = {}
    for var in daily_df.keys():
        # try:
        m = np.full(12, np.nan)
        for i in np.arange(12):
            m_values = daily_df[var].values[daily_df.index.month == i + 1]
            if np.sum(~np.isnan(m_values)) > n_days_thres:  # only save if more than n_days_thres valid days in month
                m[i] = np.nanmean(m_values)
            else:
                if var=='nep':
                    print('{} has only {} days in month #{}'.format(site,np.sum(~np.isnan(m_values)),i+1))
        d[var] = m
    month_df = pd.DataFrame.from_dict(d)
    month_df.index = np.arange(1, 12 + 1)
    month_df.index = month_df.index.set_names('Month')
    collated_dict[site]['month_df'] = month_df
    # except:
    #     print('skipping {}'.format(var))

# calculate means for different cloud conditions (hourly)
for site in collated_dict.keys():
    collated_dict[site]['cloud_diff_hourly'] = {}
    full_dict = collated_dict[site]
    daily_df = full_dict['aws_df']
    for var in ['swnet', 'lwnet', 'rnet', 'qs', 'ql', 'qc', 'qm', 'tc','rh','ws','ea','ts']:
        if var in daily_df.keys():
            m = np.full((12, 9), np.nan)
            for i in np.arange(12):
                m_values = daily_df[var].values[daily_df.index.month == i + 1]
                if np.sum(~np.isnan(m_values)) > n_days_thres * 24:  # only save if more than n_days_thres valid days in month
                    m_cloud = daily_df['nep'].values[daily_df.index.month == i + 1]
                    m[i, 0] = np.nanmean(m_values[m_cloud <= cs_thres])
                    m[i, 1] = np.nanmean(m_values[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)])
                    m[i, 2] = np.nanmean(m_values[m_cloud >= ov_thres])
                    m[i, 3] = np.sum(~np.isnan(m_values[m_cloud <= cs_thres]))
                    m[i, 4] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)]))
                    m[i, 5] = np.sum(~np.isnan(m_values[m_cloud >= ov_thres]))
                    m[i, 6] = np.nanstd(m_values[m_cloud <= cs_thres])
                    m[i, 7] = np.nanstd(m_values[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)])
                    m[i, 8] = np.nanstd(m_values[m_cloud >= ov_thres])

            var_df = pd.DataFrame(m, columns=['cs_m', 'pc_m', 'ov_m', 'cs_n', 'pc_n', 'ov_n', 'cs_std', 'pc_std', 'ov_std'])
            var_df.index = np.arange(1, 12 + 1)
            var_df.index = var_df.index.set_names('Month')
            collated_dict[site]['cloud_diff_hourly'][var] = var_df

# calculate means for different cloud conditions (daily)
for site in collated_dict.keys():
    collated_dict[site]['cloud_diff'] = {}
    full_dict = collated_dict[site]
    daily_df = full_dict['daily_df']
    for var in daily_df.keys():
        m = np.full((12, 9), np.nan)
        for i in np.arange(12):
            m_values = daily_df[var].values[daily_df.index.month == i + 1]
            if np.sum(~np.isnan(m_values)) > n_days_thres:  # only save if more than n_days_thres * 24 valid hours in month
                m_cloud = daily_df['nep'].values[daily_df.index.month == i + 1]
                m[i, 0] = np.nanmean(m_values[m_cloud <= cs_thres])
                m[i, 1] = np.nanmean(m_values[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)])
                m[i, 2] = np.nanmean(m_values[m_cloud >= ov_thres])
                m[i, 3] = np.sum(~np.isnan(m_values[m_cloud <= cs_thres]))
                m[i, 4] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)]))
                m[i, 5] = np.sum(~np.isnan(m_values[m_cloud >= ov_thres]))
                m[i, 6] = np.nanstd(m_values[m_cloud <= cs_thres])
                m[i, 7] = np.nanstd(m_values[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)])
                m[i, 8] = np.nanstd(m_values[m_cloud >= ov_thres])

        var_df = pd.DataFrame(m, columns=['cs_m', 'pc_m', 'ov_m', 'cs_n', 'pc_n', 'ov_n', 'cs_std', 'pc_std', 'ov_std'])
        var_df.index = np.arange(1, 12 + 1)
        var_df.index = var_df.index.set_names('Month')
        collated_dict[site]['cloud_diff'][var] = var_df

# calculate fraction of month in clear-sky, partial cloud and overcast from hourly data
for site in collated_dict.keys():
    collated_dict[site]['cloud_clim_hourly'] = {}
    full_dict = collated_dict[site]
    aws_df = full_dict['aws_df']
    var = 'nep'
    m = np.full((12, 6), np.nan)
    for i in np.arange(12):
        m_cloud = aws_df[var].values[aws_df.index.month == i + 1]
        if np.sum(~np.isnan(m_cloud)) > n_days_thres * 24:
            m[i, 0] = np.sum(~np.isnan(m_cloud[m_cloud <= cs_thres]))
            m[i, 1] = np.sum(~np.isnan(m_cloud[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)]))
            m[i, 2] = np.sum(~np.isnan(m_cloud[m_cloud >= ov_thres]))
            n = np.sum(~np.isnan(m_cloud))
            m[i, 3] = np.sum(~np.isnan(m_cloud[m_cloud <= cs_thres])) / n
            m[i, 4] = np.sum(~np.isnan(m_cloud[np.logical_and(m_cloud > cs_thres, m_cloud < ov_thres)])) / n
            m[i, 5] = np.sum(~np.isnan(m_cloud[m_cloud >= ov_thres])) / n
    var_df = pd.DataFrame(m, columns=['cs_n', 'pc_n', 'ov_n', 'cs_f', 'pc_f', 'ov_f'])
    var_df.index = np.arange(1, 12 + 1)
    var_df.index = var_df.index.set_names('Month')
    collated_dict[site]['cloud_clim_hourly'][var] = var_df

# calculate fraction of month in clear-sky, partial cloud and overcast from daily data
for site in collated_dict.keys():
    collated_dict[site]['cloud_clim_daily'] = {}
    full_dict = collated_dict[site]
    cloud_df = collated_dict[site]['cloud_diff']['nep']  # could take any variable but nep will include missing timesteps from relevant variables

    m = np.full((12, 6), np.nan)
    m[:, 0] = cloud_df.cs_n.values
    m[:, 1] = cloud_df.pc_n.values
    m[:, 2] = cloud_df.ov_n.values
    n = (cloud_df.ov_n.values + cloud_df.pc_n.values + cloud_df.cs_n.values)
    m[:, 3] = cloud_df.cs_n.values / n
    m[:, 4] = cloud_df.pc_n.values / n
    m[:, 5] = cloud_df.ov_n.values / n
    var_df = pd.DataFrame(m, columns=['cs_n', 'pc_n', 'ov_n', 'cs_f', 'pc_f', 'ov_f'])
    var_df.index = np.arange(1, 12 + 1)
    var_df.index = var_df.index.set_names('Month')
    collated_dict[site]['cloud_clim_daily']['nep'] = var_df

# calculate fraction of month melt occurs in clear-sky, partial cloud and overcast from hourly data
for site in collated_dict.keys():
    if 'qm' in collated_dict[site]['aws_df'].keys():
        collated_dict[site]['melt_clim_hourly'] = {}
        for cloud_var in ['nep', 'nep_av24']:
            full_dict = collated_dict[site]
            aws_df = full_dict['aws_df']
            m = np.full((12, 6), np.nan)
            for i in np.arange(12):

                m_melt = aws_df['qm'].values[aws_df.index.month == i + 1]
                m_cloud = aws_df[cloud_var].values[aws_df.index.month == i + 1]  # use daily average nep
                m_ind = np.logical_and(~np.isnan(m_cloud), ~np.isnan(m_melt)) # find times with good melt and cloud datasets
                m_melting = m_melt > 0 # turn into binary
                m_melting_nonan = m_melting[m_ind]  # take only times with good melt and cloud datasets
                m_cloud_nonan = m_cloud[m_ind]
                if len(m_melting_nonan) > n_days_thres * 24:  # use only if greater than n_days_thres days in month with good melt and cloud datasets
                    m[i, 0] = np.sum(m_melting_nonan[m_cloud_nonan <= cs_thres])
                    m[i, 1] = np.sum(m_melting_nonan[np.logical_and(m_cloud_nonan > cs_thres, m_cloud_nonan < ov_thres)])
                    m[i, 2] = np.sum(m_melting_nonan[m_cloud_nonan >= ov_thres])
                    m[i, 3] = np.sum(m_melting_nonan[m_cloud_nonan <= cs_thres]) / len(m_melting_nonan[m_cloud_nonan <= cs_thres])
                    m[i, 4] = np.sum(m_melting_nonan[np.logical_and(m_cloud_nonan > cs_thres, m_cloud_nonan < ov_thres)]) / len(
                        m_melting_nonan[np.logical_and(m_cloud_nonan > cs_thres, m_cloud_nonan < ov_thres)])
                    m[i, 5] = np.sum(m_melting_nonan[m_cloud_nonan >= ov_thres]) / len(m_melting_nonan[m_cloud_nonan >= ov_thres])
            var_df = pd.DataFrame(m, columns=['cs_n', 'pc_n', 'ov_n', 'cs_f', 'pc_f', 'ov_f'])
            var_df.index = np.arange(1, 12 + 1)
            var_df.index = var_df.index.set_names('Month')
            collated_dict[site]['melt_clim_hourly'][cloud_var] = var_df

for site in collated_dict.keys():
    if 'qm' in collated_dict[site]['aws_df'].keys():
        full_dict = collated_dict[site]
        aws_df = full_dict['aws_df']
        d = {}
        d['melt_freq'] = np.full(12, np.nan)
        for j, var in enumerate(['swnet', 'lwnet', 'rnet', 'qs', 'ql', 'qc', 'qm']):  # 'qr', 'qps',
            if var in aws_df.keys():
                m = np.full(12, np.nan)
                for i in np.arange(12):
                    m_melt = aws_df['qm'].values[aws_df.index.month == i + 1]
                    m_values = aws_df[var].values[aws_df.index.month == i + 1]
                    m_ind = np.logical_and(~np.isnan(m_values), ~np.isnan(m_melt))
                    m_melt = m_melt[m_ind]
                    m_values = m_values[m_ind]
                    if len(m_melt) > n_days_thres *24:  # only save if more than n_days_thres valid days in month
                        # try:
                        m[i] = np.mean(m_values[m_melt > 0])
                        # except:
                        #     print('')
                        d['melt_freq'][i] = np.sum(m_melt > 0) / len(m_melt)
                d[var] = m
        month_df = pd.DataFrame.from_dict(d)
        month_df.index = np.arange(1, 12 + 1)
        month_df.index = month_df.index.set_names('Month')
        collated_dict[site]['melt_month_df'] = month_df

pickle.dump(collated_dict, open(outfile.format(cs_thres,ov_thres,n_days_thres), 'wb'), -1)
