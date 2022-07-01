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
from adjustText import adjust_text
from scipy.stats import linregress
from cycler import cycler

# set breakpoints in nep to use for 'clear-sky' and 'overcast' conditions - doesn't really matter here - only use monthly average values
cs_thres = 0.20
ov_thres = 0.80
n_days_thres = 10
ind_frac_max_melt = 0.2  # fraction of maximum monthly average melt rate to use to select months for analysis

data_dir = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/'
plot_dir = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/plots/publication/revision/{}/'.format(ind_frac_max_melt)
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/collated_monthly_{}_{}_{}.pkl'.format(cs_thres, ov_thres,
                                                                                                                                n_days_thres)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

collated_dict = pickle.load(open(infile, 'rb'))

cloud_var = 'nep_av24'
for site in collated_dict.keys():

    full_dict = collated_dict[site]
    aws_df = full_dict['aws_df']
    take_months = full_dict['month_df'].qm > np.max(full_dict['month_df'].qm) * ind_frac_max_melt  # create boolean
    take_months = take_months.index.values[take_months]  # create array with integers of months to take
    ind = []
    for i in aws_df.index.month:
        ind.append(i in take_months)
    ind = np.asarray(ind)

    ind_daily = []
    for i in full_dict['daily_df'].index.month:
        ind_daily.append(i in take_months)
    ind_daily = np.asarray(ind_daily)

    # collate daily average qm in cloud bins for plotting as histograms
    m = []
    m_values = full_dict['daily_df']['qm'].values[ind_daily]
    m_cloud = full_dict['daily_df'][cloud_var].values[ind_daily]
    m.append(np.nanmean(m_values))  # store mean melt energy during melting months
    m.append(np.nanmean(m_cloud))  # store mean cloudiness during melting months

    m.append(m_values[np.logical_and(~np.isnan(m_values), m_cloud < 0.2)])
    m.append(m_values[np.logical_and(~np.isnan(m_values), np.logical_and(m_cloud >= .2, m_cloud < .4))])
    m.append(m_values[np.logical_and(~np.isnan(m_values), np.logical_and(m_cloud >= .4, m_cloud < .6))])
    m.append(m_values[np.logical_and(~np.isnan(m_values), np.logical_and(m_cloud >= .6, m_cloud < .8))])
    m.append(m_values[np.logical_and(~np.isnan(m_values), m_cloud >= 0.8)])

    collated_dict[site]['cloud_melt_boxplot_data'] = m

    # calculate average qm for all periods in selected months # don't worry about filtering by n_days_thres as all months being pooled
    collated_dict[site]['cloud_diff'] = {}
    for var in ['qm', 'swnet', 'lwnet', 'rnet', 'qs', 'ql', 'qc', 'tc', 'rh', 'ws', 'ea', 'ts', 'alb', 'swin', 'lwin', 'nep_av24']:
        if var in aws_df.keys():
            m = np.full((3, 5), np.nan)
            m_values = aws_df[var].values[ind]
            m_cloud = aws_df[cloud_var].values[ind]

            i = 0
            m[i, 0] = np.nanmean(m_values[m_cloud < 0.2])
            m[i, 1] = np.nanmean(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)])
            m[i, 2] = np.nanmean(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)])
            m[i, 3] = np.nanmean(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)])
            m[i, 4] = np.nanmean(m_values[m_cloud >= 0.8])

            i = 1
            m[i, 0] = np.nanstd(m_values[m_cloud < 0.2])
            m[i, 1] = np.nanstd(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)])
            m[i, 2] = np.nanstd(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)])
            m[i, 3] = np.nanstd(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)])
            m[i, 4] = np.nanstd(m_values[m_cloud >= 0.8])

            i = 2
            m[i, 0] = np.sum(~np.isnan(m_values[m_cloud < 0.2]))
            m[i, 1] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)]))
            m[i, 2] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)]))
            m[i, 3] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)]))
            m[i, 4] = np.sum(~np.isnan(m_values[m_cloud >= 0.8]))

            var_df = pd.DataFrame(np.transpose(m), columns=['mean', 'std', 'num'])
            var_df.index = np.arange(0.1, 1.1, 0.2).round(decimals=1)
            var_df.index = var_df.index.set_names(cloud_var)
            collated_dict[site]['cloud_diff'][var] = var_df

    # calculate average for all periods with melt in selected months # don't worry about filtering by n_days_thres as all months being pooled
    collated_dict[site]['cloud_diff_during_melt'] = {}
    for var in ['qm', 'swnet', 'lwnet', 'rnet', 'qs', 'ql', 'qc', 'tc', 'rh', 'ws', 'ea', 'ts', 'alb', 'swin', 'lwin']:
        if var in aws_df.keys():
            m = np.full((3, 5), np.nan)
            melt = aws_df['qm'].values
            ind_melt = np.logical_and(ind, melt > 0)
            m_values = aws_df[var].values[ind_melt]
            m_cloud = aws_df[cloud_var].values[ind_melt]

            i = 0
            m[i, 0] = np.nanmean(m_values[m_cloud < 0.2])
            m[i, 1] = np.nanmean(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)])
            m[i, 2] = np.nanmean(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)])
            m[i, 3] = np.nanmean(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)])
            m[i, 4] = np.nanmean(m_values[m_cloud >= 0.8])

            i = 1
            m[i, 0] = np.nanstd(m_values[m_cloud < 0.2])
            m[i, 1] = np.nanstd(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)])
            m[i, 2] = np.nanstd(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)])
            m[i, 3] = np.nanstd(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)])
            m[i, 4] = np.nanstd(m_values[m_cloud >= 0.8])

            i = 2
            m[i, 0] = np.sum(~np.isnan(m_values[m_cloud < 0.2]))
            m[i, 1] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)]))
            m[i, 2] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)]))
            m[i, 3] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)]))
            m[i, 4] = np.sum(~np.isnan(m_values[m_cloud >= 0.8]))

            var_df = pd.DataFrame(np.transpose(m), columns=['mean', 'std', 'num'])
            var_df.index = np.arange(0.1, 1.1, 0.2).round(decimals=1)
            var_df.index = var_df.index.set_names(cloud_var)
            collated_dict[site]['cloud_diff_during_melt'][var] = var_df

    # calculate frequency of melt in selected cloud conditions in selected months # don't worry about filtering by n_days_thres as all months being pooled
    collated_dict[site]['melt_freq_all_time'] = {}
    m = np.full((2, 5), np.nan)
    m_values = aws_df['qm'].values[ind]
    m_cloud = aws_df[cloud_var].values[ind]

    i = 0
    m[i, 0] = np.sum(m_values[m_cloud < 0.2] > 0) / np.sum(~np.isnan(m_values[m_cloud < 0.2]))
    m[i, 1] = np.sum(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)] > 0) / np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)]))
    m[i, 2] = np.sum(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)] > 0) / np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)]))
    m[i, 3] = np.sum(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)] > 0) / np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)]))
    m[i, 4] = np.sum(m_values[m_cloud >= 0.8] > 0) / np.sum(~np.isnan(m_values[m_cloud >= 0.8]))

    i = 1
    m[i, 0] = np.sum(~np.isnan(m_values[m_cloud < 0.2]))
    m[i, 1] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .2, m_cloud < .4)]))
    m[i, 2] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .4, m_cloud < .6)]))
    m[i, 3] = np.sum(~np.isnan(m_values[np.logical_and(m_cloud >= .6, m_cloud < .8)]))
    m[i, 4] = np.sum(~np.isnan(m_values[m_cloud >= 0.8]))

    var_df = pd.DataFrame(np.transpose(m), columns=['freq', 'num'])
    var_df.index = np.arange(0.1, 1.1, 0.2).round(decimals=1)
    var_df.index = var_df.index.set_names(cloud_var)
    collated_dict[site]['melt_freq_all_time'] = var_df

####

lats = []
sites_in_dict = []
for site in collated_dict.keys():
    lats.append(collated_dict[site]['aws_loc'].latitude)
    sites_in_dict.append(site)

# pos = np.argsort(lats)[::-1]
# sites_to_plot = [sites_in_dict[i] for i in pos]

sites_to_plot = ['langfjordjokelen', 'qasi', 'storbreen', 'midtdalsbreen', 'nordic', 'conrad_abl', 'conrad_acc', 'morteratsch',
                 'chhota_shigri', 'yala', 'naulek', 'mera_summit', 'kersten', 'zongo', 'guanaco', 'brewster']

site_labels = {
    'langfjordjokelen': 'lang',
    'qasi': 'qasi',
    'storbreen': 'stor',
    'midtdalsbreen': 'midt',
    'nordic': 'nord',
    'conrad_abl': 'cabl',
    'conrad_acc': 'cacc',
    'morteratsch': 'mort',
    'chhota_shigri': 'chho',
    'yala': 'yala',
    'naulek': 'naul',
    'mera_summit': 'mera',
    'kersten': 'kers',
    'zongo': 'zong',
    'guanaco': 'guan',
    'brewster': 'brew'
}

# take subset of colors from plt.get_cmap('tab20b',20).colors that are distinct with color blindness. tested using https://www.color-blindness.com/coblis-color-blindness-simulator/
colors = [
    [0.09019608, 0.74509804, 0.81176471, 1.],
    [0.7372549, 0.74117647, 0.13333333, 1.],
    [0.17254902, 0.62745098, 0.17254902, 1.],
    [0.54901961, 0.3372549, 0.29411765, 1.],
]

# define cycle of markers and line colors
cc = (cycler(color=colors) * cycler(markerstyle=['P', 'X', '*', 'd']))  # ['D','o','^','s'])) ['P','X', '*', 'd']
cstyle = []
for d in cc:
    cstyle.append(d)

# plot altitude and latitude of sites
fig, ax = plt.subplots(figsize=[6, 4])
text = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        alt = full_dict['aws_loc'].altitude
        lat = full_dict['aws_loc'].latitude
        ax.scatter(alt, lat, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        h = ax.annotate(site_labels[site].upper(),  # text to display
                        (alt, lat),  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(2, 2),  # distance from text to points (x,y)
                        ha='right', fontsize=10)  # fontweight='bold', color='b')
        if lat < 0:
            ax.scatter(alt, -1 * lat, 100, marker=cstyle[ii]['markerstyle'], edgecolors=cstyle[ii]['color'], facecolors='none')
        text.append(h)
plt.xlabel('Altitude (metres a.s.l)')
plt.ylabel('Latitude (°N)')
plt.axhline(0, color='k', linestyle=':')
fig.tight_layout()
adjust_text(text)
plt.savefig(plot_dir + 'Fig2.png'.format(ind_frac_max_melt), dpi=600, format='png')
plt.savefig(plot_dir + 'Fig2.pdf'.format(ind_frac_max_melt), dpi=600, format='pdf')

plt.rcParams.update({'font.size': 8})

# plot average cloud effects by site attributes
fig, axs = plt.subplots(4, 5, figsize=[10, 8])
axs = axs.ravel()
ax = axs[0]
plt.sca(ax)
plt.ylabel(r'$Q_M$ CE ($W m^{-2}$)')
plt.title('(a)')
x = []
y = []
plt.xlabel('Altitude (m)')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = alt = full_dict['aws_loc'].altitude
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(a) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[1]
plt.sca(ax)
x = []
y = []
plt.xlabel('Absolute latitude (°)')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.abs(full_dict['aws_loc'].latitude)
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(b) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[2]
plt.sca(ax)
plt.xlabel('$T_a$ (°C)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['tc']['mean'])  # ta averaged equally across cloudiness bins
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(c) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[3]
plt.sca(ax)
plt.xlabel('RH (%)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['rh']['mean'])  # rh averaged equally across cloudiness bins
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(d) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[4]
plt.sca(ax)
plt.xlabel(r'$N_\epsilon$')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = full_dict['cloud_melt_boxplot_data'][1]  # average cloudiness in selected months
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(e) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[5]
plt.sca(ax)
plt.ylabel(r'$Q_M$ CE ($W m^{-2}$)')
plt.xlabel(r'$WS$ ($m s^{-1}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['ws']['mean'])
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(f) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[6]
plt.sca(ax)
plt.xlabel(r'$Q_S$ ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['qs']['mean'])
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(g) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[7]
plt.sca(ax)
plt.xlabel(r'$Q_L$ ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['ql']['mean'])
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(h) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[8]
plt.sca(ax)
plt.xlabel(r'$SWin$ ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['swin']['mean'])
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(i) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[9]
plt.sca(ax)
plt.xlabel(r'$LWin$ ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['lwin']['mean'])
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(j) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[10]
plt.sca(ax)
plt.ylabel(r'$Q_M$ CE ($W m^{-2}$)')
plt.xlabel(r'$SWin$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['swin']['mean']) - full_dict['cloud_diff']['swin']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(k) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[11]
plt.sca(ax)
plt.xlabel(r'$LWin$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['lwin']['mean']) - full_dict['cloud_diff']['lwin']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(l) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[12]
plt.sca(ax)
plt.xlabel(r'$SWin$ + $LWin$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['swin']['mean']) - full_dict['cloud_diff']['swin']['mean'][0.1] + np.nanmean(
            full_dict['cloud_diff']['lwin']['mean']) - full_dict['cloud_diff']['lwin']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(m) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[13]
plt.sca(ax)
plt.xlabel(r'Albedo')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['alb']['mean'])  # albedo averaged equally across cloudiness bins
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(n) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[14]
plt.sca(ax)
plt.xlabel(r'$Rnet$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['rnet']['mean']) - full_dict['cloud_diff']['rnet']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(o) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[15]
plt.sca(ax)
plt.ylabel(r'$Q_M$ CE ($W m^{-2}$)')
plt.xlabel(r'$Q_S$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['qs']['mean']) - full_dict['cloud_diff']['qs']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(p) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[16]
plt.sca(ax)
plt.xlabel(r'$Q_L$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['ql']['mean']) - full_dict['cloud_diff']['ql']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(q) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[17]
plt.sca(ax)
plt.xlabel(r'$Q_L$ +$Q_S$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['ql']['mean']) - full_dict['cloud_diff']['ql']['mean'][0.1] + np.nanmean(
            full_dict['cloud_diff']['qs']['mean']) - full_dict['cloud_diff']['qs']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['qm']['mean']) - full_dict['cloud_diff']['qm']['mean'][0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(r) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[19]
plt.sca(ax)
plt.ylabel(r'$Rnet$ CE ($W m^{-2}$)')
plt.xlabel(r'$Q_L$ +$Q_S$ CE ($W m^{-2}$)')
x = []
y = []
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        lw = np.nanmean(full_dict['cloud_diff']['ql']['mean']) - full_dict['cloud_diff']['ql']['mean'][0.1] + np.nanmean(
            full_dict['cloud_diff']['qs']['mean']) - full_dict['cloud_diff']['qs']['mean'][0.1]
        sw = (np.nanmean(full_dict['cloud_diff']['rnet']['mean']) - full_dict['cloud_diff']['rnet']['mean'][
            0.1])
        ax.scatter(lw, sw, 100, label=site_labels[site].upper(), marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])
        x.append(lw)
        y.append(sw)
plt.title('(s) r={},p={}'.format(linregress(x, y)[2].round(2), linregress(x, y)[3].round(2)))

ax = axs[18]
plt.sca(ax)
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.scatter(-1, -1, label=site_labels[site].upper(),
                    marker=cstyle[ii]['markerstyle'], color=cstyle[ii]['color'])

plt.yticks()
plt.xticks()
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.axis('off')
plt.legend(ncol=2, loc='center')
fig.tight_layout()
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig12.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig12.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'cloud effect qm from bins 4by5 cb3 {}.png'.format(cloud_var), dpi=300, bbox_inches='tight')

# create cylce of linestyles with same colors as points
cd = (cycler(color=colors) * cycler(linestyle=[':', '--', '-.', '-']))  # ['D','o','^','s']))
clstyle = []
for d in cd:
    clstyle.append(d)

# plot incoming rad with respect to clear-sky
fig, axs = plt.subplots(2, 3, figsize=[10, 7.5])
axs = axs.ravel()
plt.sca(axs[0])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.ylabel(r'Radiation flux ($W m^{-2}$)')
plt.title('(a) $SWin$')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['swin']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[1])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(b) $LWin$')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['lwin']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[2])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(c) $SWin$ + $LWin$')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['swin']['mean'] + full_dict['cloud_diff']['lwin']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
lg = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
plt.sca(axs[3])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel(r'Radiation flux ($W m^{-2}$)')
plt.title('(d) $SWin$ cloud effect')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['swin']['mean'] - full_dict['cloud_diff']['swin']['mean'][0.1], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[4])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(e) $LWin$ cloud effect')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['lwin']['mean'] - full_dict['cloud_diff']['lwin']['mean'][0.1], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[5])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(f) $SWin$ + $LWin$ cloud effect')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot((full_dict['cloud_diff']['swin']['mean'] - full_dict['cloud_diff']['swin']['mean'][0.1]) + (
                full_dict['cloud_diff']['lwin']['mean'] - full_dict['cloud_diff']['lwin']['mean'][0.1]), label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig6.pdf', dpi=600, format='pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig6.png', dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'allwave incoming by cloud 6 panel cb3 {}.png'.format(cloud_var), dpi=300, bbox_extra_artists=(lg,), bbox_inches='tight')

# plot rad with respect to clear-sky
fig, axs = plt.subplots(2, 3, figsize=[10, 7.5])
axs = axs.ravel()
plt.sca(axs[0])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.ylabel(r'Radiation flux ($W m^{-2}$)')
plt.title('(a) $SWnet$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['swnet']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[1])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(b) $LWnet$')
plt.axhline(1, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['lwnet']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[2])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(c) $Rnet$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['rnet']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
lg = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
plt.sca(axs[3])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel(r'Radiation flux ($W m^{-2}$)')
plt.title('(d) $SWnet$ cloud effect')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['swnet']['mean'] - full_dict['cloud_diff']['swnet']['mean'][0.1], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[4])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(e) $LWnet$ cloud effect')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['lwnet']['mean'] - full_dict['cloud_diff']['lwnet']['mean'][0.1], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[5])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(f) $Rnet$ cloud effect')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['rnet']['mean'] - full_dict['cloud_diff']['rnet']['mean'][0.1], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig7.pdf', dpi=600, format='pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig7.png', dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'Net rad terms by cloud 6 panel {}.png'.format(cloud_var), dpi=300, bbox_extra_artists=(lg,), bbox_inches='tight')

# plot melt frequency and fraction of time in each cloud condition
fig, axs = plt.subplots(1, 2, figsize=[7, 4])
plt.sca(axs[0])
plt.ylim([0, 100])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel('% of hours with surface melt')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['melt_freq_all_time']['freq'] * 100, label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.grid()
plt.sca(axs[1])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel('Ratio to clear-sky conditions')
plt.axhline(1, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        dat = full_dict['melt_freq_all_time']['freq'] / full_dict['melt_freq_all_time']['freq'][0.1]
        if site == 'guanaco':
            dat.values[1:] = np.nan
        if site == 'mera_summit':
            for i in [0.5, 0.7, 0.9]:
                plt.annotate(dat[i].round(decimals=2),  # text to display
                             (i, 3.03),  # this is the point to label
                             # textcoords="offset points",  # how to position the text
                             # xytext=(2, 2),  # distance from text to points (x,y)
                             ha='right', fontsize=10, color=clstyle[ii]['color'])  # fontweight='bold', color='b')
        plt.plot(dat, label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.ylim(top=3.2)
plt.grid()
lg = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig9.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig9.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'melt and normalised melt freqency by cloud with limit A {}.png'.format(cloud_var), dpi=300, bbox_extra_artists=(lg,),
                bbox_inches='tight')

# plot average melt energy by cloud fraction, with normalised
fig, axs = plt.subplots(1, 2, figsize=[7, 4])
plt.sca(axs[0])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel(r'$Q_M$ ($W m^{-2}$)')
plt.axhline(1, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['qm']['mean'], label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.grid()
plt.sca(axs[1])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel(r'Ratio to clear-sky conditions')
plt.axhline(1, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        dat = full_dict['cloud_diff']['qm']['mean'] / full_dict['cloud_diff']['qm']['mean'][0.1]
        if ind_frac_max_melt == .8:
            if site in ['guanaco', 'kersten']:
                dat.values[1:] = np.nan
        else:
            if site == 'guanaco':
                dat.values[1:] = np.nan
            if site == 'mera_summit':
                for i in [0.3, 0.5, 0.7, 0.9]:
                    plt.annotate(dat[i].round(decimals=2),  # text to display
                                 (i, 2.03),  # this is the point to label
                                 # textcoords="offset points",  # how to position the text
                                 # xytext=(2, 2),  # distance from text to points (x,y)
                                 ha='right', fontsize=10, color=clstyle[ii]['color'])
        plt.plot(dat, label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
if ind_frac_max_melt != .8:
    plt.ylim(bottom=.4, top=2.2)
    plt.yticks(np.arange(0.4, 2.2, 0.2))
plt.grid()
lg = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig10.pdf', dpi=600, format='pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig10.png', dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
elif '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.8':
    plt.savefig(plot_dir + 'Fig13.pdf', dpi=600, format='pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig13.png', dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'qm wrt cs{}.png'.format(cloud_var), dpi=300, bbox_extra_artists=(lg,), bbox_inches='tight')

# plot SEB terms as fraction of Qm during melting periods only
fig, axs = plt.subplots(2, 3, figsize=[10, 7.5])
axs = axs.ravel()
plt.sca(axs[0])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(a) $SWnet$')
plt.ylabel(r'Fraction of $Q_M$')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['swnet']['mean'] / full_dict['cloud_diff_during_melt']['qm']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[1])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(b) $LWnet$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['lwnet']['mean'] / full_dict['cloud_diff_during_melt']['qm']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[2])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(c) $Rnet$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['rnet']['mean'] / full_dict['cloud_diff_during_melt']['qm']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
lg = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
plt.sca(axs[3])
plt.ylabel(r'Fraction of $Q_M$')
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(d) $Q_S$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['qs']['mean'] / full_dict['cloud_diff_during_melt']['qm']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[4])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(e) $Q_L$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['ql']['mean'] / full_dict['cloud_diff_during_melt']['qm']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[5])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(f) $LWnet+Q_S+Q_L$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot((full_dict['cloud_diff_during_melt']['lwnet']['mean'] + full_dict['cloud_diff_during_melt']['qs']['mean'] +
                  full_dict['cloud_diff_during_melt']['ql']['mean']) / full_dict['cloud_diff_during_melt']['qm']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig11.pdf', dpi=600, format='pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig11.png', dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'SEB terms during melt as fraction of qm by cloud {}.png'.format(cloud_var), dpi=300, bbox_extra_artists=(lg,), bbox_inches='tight')

# plot SEB terms during melting periods only
fig, axs = plt.subplots(2, 4, figsize=[10, 7.5])
axs = axs.ravel()
plt.sca(axs[0])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(a) $SWnet$')
plt.ylabel(r'Energy flux ($W m^{-2}$)')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['swnet']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[1])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(b) $LWnet$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['lwnet']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[2])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.title('(c) $Rnet$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['rnet']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
lg = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
plt.sca(axs[3])
plt.ylabel(r'Energy flux ($W m^{-2}$)')
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(d) $Q_S$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['qs']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[4])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(e) $Q_L$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff_during_melt']['ql']['mean'], label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[5])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(f) $LWnet+Q_S+Q_L$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot((full_dict['cloud_diff_during_melt']['lwnet']['mean'] + full_dict['cloud_diff_during_melt']['qs']['mean'] +
                  full_dict['cloud_diff_during_melt']['ql']['mean']), label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[6])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.title('(f) $Q_S+Q_L$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot((full_dict['cloud_diff_during_melt']['qs']['mean'] +
                  full_dict['cloud_diff_during_melt']['ql']['mean']), label=site_labels[site].upper(),
                 linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'FigA5.pdf', dpi=600, format='pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(plot_dir + 'FigA5.png', dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'SEB terms during melt qm by cloud {}.png'.format(cloud_var), dpi=300, bbox_extra_artists=(lg,), bbox_inches='tight')

# plot meteorology by cloud condition
fig, axs = plt.subplots(1, 4, figsize=[11.5, 4])
plt.sca(axs[0])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel('$^{o}C$')
plt.title('(a) $T_a$')
plt.axhline(0, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['tc']['mean'], label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[1])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel('$m s^{-1}$')
plt.title('(b) wind speed')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['ws']['mean'], label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[2])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel('$hPa$')
plt.title('(c) $e_a$')
plt.axhline(6.15, color='k', linestyle='--')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['ea']['mean'], label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
plt.sca(axs[3])
plt.xticks(np.linspace(0.1, 0.9, 5))
plt.xlabel(r'$N_\epsilon$ bin centre')
plt.ylabel(r'%')
plt.title('(d) $RH$')
for ii, site in enumerate(sites_to_plot):
    if site in sites_to_plot:
        full_dict = collated_dict[site]
        plt.plot(full_dict['cloud_diff']['rh']['mean'], label=site_labels[site].upper(), linestyle=clstyle[ii]['linestyle'], color=clstyle[ii]['color'])
lg = plt.legend(bbox_to_anchor=(1.5, 0.5), loc='center right')
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig8.pdf', dpi=600, format='pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig8.png', dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'met by cloud {}.png'.format(cloud_var), dpi=300, bbox_extra_artists=(lg,), bbox_inches='tight')
