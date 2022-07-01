"""
plotting relationships between daily cloud metrics for different site
Jono Conway
jono.conway@niwa.co.nz
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from calc_cloud_metrics import find_cs_from_lw

# set breakpoints in nep to use for 'clear-sky' and 'overcast' conditions
cs_thres = 0.2
ov_thres = 0.8
n_days_thres = 10
ind_frac_max_melt = .8

data_dir = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/'
plot_dir = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/plots/publication/revision/{}_{}_{}_{}/'.format(cs_thres, ov_thres, n_days_thres,ind_frac_max_melt)
infile = 'C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/collated/collated_monthly_{}_{}_{}.pkl'.format(cs_thres, ov_thres,
                                                                                                                                n_days_thres)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

collated_dict = pickle.load(open(infile, 'rb'))


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

# take subset of colors from plt.get_cmap('tab20b',20).colors that are distinct with  color blindness. tested using https://www.color-blindness.com/coblis-color-blindness-simulator/
colors = [
    [0.12156863, 0.46666667, 0.70588235, 1.],
    [0.09019608, 0.74509804, 0.81176471, 1.],
    [0.58039216, 0.40392157, 0.74117647, 1.],
    [0.7372549, 0.74117647, 0.13333333, 1.],
    [0.17254902, 0.62745098, 0.17254902, 1.],
    [0.54901961, 0.3372549, 0.29411765, 1.],
    [1., 0.73333333, 0.47058824, 1.],
]

fig, axs = plt.subplots(4, 4, sharey=True, sharex=True, figsize=(9, 9))
for i, site in enumerate(sites_to_plot):
    full_dict = collated_dict[site]
    take_months = full_dict['month_df'].qm > np.max(full_dict['month_df'].qm) * ind_frac_max_melt  # create boolean
    blue = take_months.values

    cloud_df = collated_dict[site]['cloud_clim_hourly']['nep']  # could take any variable but nep will include missing timesteps from relevant variables
    cs_f = cloud_df.cs_f.values.copy()
    pc_f = cloud_df.pc_f.values.copy()
    ov_f = cloud_df.ov_f.values.copy()
    # make sure plots correctly when not all months have data
    ind = np.logical_and(np.isnan(cs_f), np.isnan(pc_f), np.isnan(ov_f))
    cs_f[ind] = 0
    ov_f[ind] = 0
    pc_f[ind] = 0
    fig.sca(axs.ravel()[i])
    plt.bar(np.arange(1, 13), cs_f, facecolor=[.8, .8, .8], label='clear-sky')
    plt.bar(np.arange(1, 13), pc_f, bottom=cs_f, facecolor=[.6, .6, .6], label='partly-cloudy')
    plt.bar(np.arange(1, 13), ov_f, bottom=cs_f + pc_f, facecolor=[.3, .3, .3], label='overcast')

    plt.bar(np.arange(1, 13)[blue], cs_f[blue], facecolor=[0.741, 0.843, 0.906], label='clear-sky')
    plt.bar(np.arange(1, 13)[blue], pc_f[blue], bottom=cs_f[blue], facecolor=[0.42 , 0.682, 0.839], label='partly-cloudy')
    plt.bar(np.arange(1, 13)[blue], ov_f[blue], bottom=cs_f[blue] + pc_f[blue], facecolor=[0.031, 0.318, 0.612], label='overcast')#0.129, 0.443, 0.71

    plt.xticks(np.arange(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    plt.ylim(0, 1)
    if i >= 12:
        plt.xlabel('Month')
    if i % 4 == 0:
        plt.ylabel('Fraction of hours')
    plt.title(site_labels[site].upper())

plt.tight_layout()
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres,ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'FigA4.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'FigA4.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'all monthly hourly cloud fraction with melt.png', dpi=300, format='png', bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(4, 4, sharey=True, sharex=True, figsize=(9, 9))
for i, site in enumerate(sites_to_plot):
    full_dict = collated_dict[site]
    take_months = full_dict['month_df'].qm > np.max(full_dict['month_df'].qm) * ind_frac_max_melt  # create boolean
    blue = take_months.values
    cloud_df = collated_dict[site]['cloud_clim_daily']['nep']  # could take any variable but nep will include missing timesteps from relevant variables
    cs_f = cloud_df.cs_f.values.copy()
    pc_f = cloud_df.pc_f.values.copy()
    ov_f = cloud_df.ov_f.values.copy()
    # make sure plots correctly when not all months have data
    ind = np.logical_and(np.isnan(cs_f), np.isnan(pc_f), np.isnan(ov_f))
    cs_f[ind] = 0
    ov_f[ind] = 0
    pc_f[ind] = 0
    fig.sca(axs.ravel()[i])
    plt.bar(np.arange(1, 13), cs_f, facecolor=[.8, .8, .8], label='clear-sky')
    plt.bar(np.arange(1, 13), pc_f, bottom=cs_f, facecolor=[.6, .6, .6], label='partly-cloudy')
    plt.bar(np.arange(1, 13), ov_f, bottom=cs_f + pc_f, facecolor=[.3, .3, .3], label='overcast')

    plt.bar(np.arange(1, 13)[blue], cs_f[blue], facecolor=[0.741, 0.843, 0.906], label='clear-sky')
    plt.bar(np.arange(1, 13)[blue], pc_f[blue], bottom=cs_f[blue], facecolor=[0.42 , 0.682, 0.839], label='partly-cloudy')
    plt.bar(np.arange(1, 13)[blue], ov_f[blue], bottom=cs_f[blue] + pc_f[blue], facecolor=[0.031, 0.318, 0.612], label='overcast')#0.129, 0.443, 0.71

    plt.xticks(np.arange(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    plt.ylim(0, 1)
    if i >= 12:
        plt.xlabel('Month')
    if i % 4 == 0:
        plt.ylabel('Fraction of days')
    plt.title(site_labels[site].upper())
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres,ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig5.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig5.png', dpi=600, format='png', bbox_inches='tight')
elif '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.8':
    plt.savefig(plot_dir + 'FigA6.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'FigA6.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'all monthly daily cloud fraction with melting months.png', dpi=300, format='png', bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(4, 4, sharey=True, sharex=True, figsize=(9, 9))
for i, site in enumerate(sites_to_plot):
    aws_df = collated_dict[site]['aws_df']
    ind_nonan = np.logical_and(~np.isnan(aws_df.lwin.values), ~np.isnan(aws_df.ea.values), ~np.isnan(aws_df.tk.values))
    ind_nonan_rh80 = np.logical_and(ind_nonan, (aws_df.rh.values < 80))
    ind10 = find_cs_from_lw(aws_df.lwin.values[ind_nonan_rh80], aws_df.ea.values[ind_nonan_rh80], aws_df.tk.values[ind_nonan_rh80])
    fig.sca(axs.ravel()[i])
    plt.scatter(aws_df.ea / aws_df.tk, aws_df.eeff, .05, 'c', label='ep_eff all points')
    plt.scatter(aws_df.ea[ind_nonan_rh80][ind10] / aws_df.tk[ind_nonan_rh80][ind10], aws_df.eeff[ind_nonan_rh80][ind10], .05, 'b',
                label='ep_eff top 10% across 30 ea/tk bins')
    ecs = np.poly1d([-9.63601916e-06, 2.35163370e-01])(collated_dict[site]['aws_loc'].altitude) + collated_dict[site]['config']['b'] * (
            np.linspace(0.001, 0.05, 50) * 100) ** (1 / 7)
    plt.plot(np.linspace(0.001, 0.05, 50), ecs, 'k', label='ep_cs')
    plt.plot(np.linspace(0.001, 0.05, 50), ecs + (1 - ecs) * .2, 'k-.', label='ep_cs')
    plt.ylim(0.2, 1.2)
    plt.xlim(0, 0.05)
    if i >= 12:
        plt.xlabel(r'$e_a/T_{aK}$ (hPa/K)')
    if i % 4 == 0:
        plt.ylabel(r'$\varepsilon_{eff}$')
    plt.title(site_labels[site].upper())
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres, ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'FigA3.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'FigA3.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'all eeff.png', dpi=300, format='png', bbox_inches='tight')
plt.close()

min_c = 1  # minimum number of points in bin to plot in contour
fig, axs = plt.subplots(4, 4, sharey=True, sharex=True, figsize=(9, 9))
for i, site in enumerate(sites_to_plot):
    aws_df = collated_dict[site]['aws_df']
    plt.figure()
    [h, xedges, yedges, imagemesh] = plt.hist2d(aws_df.ea / aws_df.tk, aws_df.eeff, bins=[np.linspace(0, 0.05, 51), np.linspace(0.2, 1.2, 51)])
    plt.close()
    fig.sca(axs.ravel()[i])
    h2 = plt.contourf(xedges[1:] - np.diff(xedges)[0] / 2, yedges[1:] - np.diff(yedges)[0] / 2, h.transpose(), levels=np.linspace(min_c, np.nanmax(h), 10),
                      cmap=plt.cm.inferno_r)
    ecs = np.poly1d([-9.63601916e-06, 2.35163370e-01])(collated_dict[site]['aws_loc'].altitude) + collated_dict[site]['config']['b'] * (
            np.linspace(0.001, 0.05, 50) * 100) ** (1 / 7)
    plt.plot(np.linspace(0.001, 0.05, 50), ecs, 'k', linewidth=0.5, label='ep_cs')
    plt.plot(np.linspace(0.001, 0.05, 50), ecs + (1 - ecs) * .2, 'k-.', linewidth=0.5, label='ep_cs')
    plt.plot(np.linspace(0.001, 0.05, 50), ecs + (1 - ecs) * .8, 'k-.', linewidth=0.5, label='ep_ov')
    plt.plot(np.linspace(0.001, 0.05, 50), np.ones(ecs.shape), 'k-', linewidth=0.5, label='ep_ov')
    plt.ylim(0.2, 1.2)
    plt.xlim(0, 0.05)
    if i >= 12:
        plt.xlabel(r'$e_a/T_{aK}$ (hPa/K)')
    if i % 4 == 0:
        plt.ylabel(r'$\varepsilon_{eff}$')
    plt.title(site_labels[site].upper())
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres,ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'Fig4.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'Fig4.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'all eeff contour {}.png'.format(min_c), dpi=300, format='png', bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(4, 4, sharey=True, sharex=True, figsize=(9, 9))
for i, site in enumerate(sites_to_plot):
    # fig, ax = plt.subplots()
    fig.sca(axs.ravel()[i])
    ax = axs.ravel()[i]
    ax_rh = plt.twinx(ax)
    # colors = plt.cm.tab10
    full_dict = collated_dict[site]
    for j, var in enumerate(['tc', 'ts', 'ea', 'rh', 'ws','alb']):
        month_df = full_dict['month_df']
        if var in ['rh', 'alb']:
            if var == 'alb':
                ax_rh.plot(month_df[var].values *100, '-x', color=colors[j], label=var)
            else:
                ax_rh.plot(month_df[var].values, '-x', color=colors[j], label=var)
        else:
            ax.plot(month_df[var].values, '-*', color=colors[j], label=var)
    ax.set_xticks(np.arange(12))
    ax_rh.set_xticks(np.arange(12))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax_rh.set_ylim([0, 100])
    ax_rh.set_yticks(np.arange(0, 120, 20))
    ax_rh.set_yticklabels('')
    if i >= 12:
        ax.set_xlabel('Month')
    if i % 4 == 0:
        ax.set_ylabel(r'°C, $hPa$, $ms^{-1}$')
        # ax.set_ylabel(r'$T_a$ (°C),$T_s$ (°C), $e_a$ (hPa), $ws (ms^{-1})$')
            #', '.join(['tc', 'ts', 'ea', 'ws']))
    if i % 4 == 3:
        ax_rh.set_yticklabels(np.arange(0, 120, 20))
        ax_rh.set_ylabel(r'$RH$, albedo (%)')
    if i == 4:
        ax.legend([r'$T_a$',r'$T_s$', r'$e_a$', r'$WS$'], loc='upper left',  facecolor='none')#frameon=False,
        ax_rh.legend([r'$RH$', 'alb'],loc='lower left', facecolor='none')#frameon=False,
    plt.title(site_labels[site].upper())
plt.tight_layout()
if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres,ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'FigA1.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'FigA1.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'all monthly met cb.png', dpi=300, format='png')  # bbox_extra_artists=(lg,),,  bbox_inches='tight'
plt.close()

fig, axs = plt.subplots(4, 4, sharey=True, sharex=True, figsize=(9, 9))
for i, site in enumerate(sites_to_plot):
    # fig, ax = plt.subplots()
    fig.sca(axs.ravel()[i])
    ax = axs.ravel()[i]
    # colors = plt.cm.tab10
    full_dict = collated_dict[site]
    month_df = full_dict['month_df']
    for j, var in enumerate(['swnet', 'lwnet', 'rnet', 'qs', 'ql', 'qm', 'qc']):  # 'qr', 'qps',
        if var in month_df.keys():
            ax.plot(month_df[var].values, '-*', color=colors[j], label=var)
        elif i == 4:# make hidden point to plot missing variables in lengend
            ax.plot(-1,0,'-*', color=colors[j],label=var)
    # plt.ylim(0, 1.2)
    plt.xlim(-0.5,11.5)
    ax.set_xticks(np.arange(12))

    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    if i >= 12:
        ax.set_xlabel('Month')
    if i % 4 == 0:
        ax.set_ylabel(r'Flux ($W m^{-2}$)')
    if i == 4:
        l = ax.legend(['$SWnet$', '$LWnet$', '$Rnet$', '$Q_S$', '$Q_L$',  '$Q_M$','$Q_C$',],loc='center left', facecolor='none',frameon=False)

    plt.title(site_labels[site].upper())
plt.tight_layout()

if '{}_{}_{}_{}'.format(cs_thres, ov_thres, n_days_thres,ind_frac_max_melt) == '0.2_0.8_10_0.2':
    plt.savefig(plot_dir + 'FigA2.png', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(plot_dir + 'FigA2.png', dpi=600, format='png', bbox_inches='tight')
else:
    plt.savefig(plot_dir + 'all monthly fluxes cb.png', dpi=300, format='png')  # bbox_extra_artists=(lg,),,  bbox_inches='tight'
plt.close()

