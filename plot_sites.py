"""
code to plot map of sites
Jono Conway
jono.conway@niwa.co.nz
"""
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from adjustText import adjust_text

fname = r"C:\Users\conwayjp\OneDrive - NIWA\projects\MarsdenFS2018\Obj1\RGI\00_rgi60_30-30grid.tif"
img = plt.imread(fname)
img2 = img / 255.0
ind = (img2[:, :, 2] == 0)  # find non-glacier points
img2[ind, 3] = 0.  # set transparency to 0 for non-glacier points

# set up dictionary with
site_list = np.genfromtxt("C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/data_sites_shortnames.txt", usecols=(1, 2, 3),
                          skip_header=1)  # ,dtype="|S12,f4,f4,f4"
keys = np.genfromtxt("C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/data_sites_shortnames.txt", usecols=(0), dtype=(str), skip_header=1)
site_info = dict(zip(keys, site_list))

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
crs = ccrs.PlateCarree()
ax.set_global()  # make the map global rather than have it zoom in to the extents of any plotted data
ax.stock_img()
ax.imshow(img2, origin='upper', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])
text = []
sites_to_plot = ['Lang', 'Qasi', 'Stor', 'Midt', 'Nord', 'Conr', 'Mort', 'Chho', 'Yala',
                  'Mera', 'Kers', 'Zong', 'Guan', 'Brew'] #'Naul',

for site in sites_to_plot:
    ax.plot(site_info[site][1], site_info[site][0], '*b', markersize=10, transform=crs, label=site)
    h = ax.annotate(site.upper(),  # text to display
                (site_info[site][1], site_info[site][0]), xycoords=crs._as_mpl_transform(ax),  # this is the point to label
                textcoords="offset points",  # how to position the text
                xytext=(-2, 2),  # distance from text to points (x,y)
                ha='right', fontsize=12, fontweight='bold',color='b')
    text.append(h)
fig.tight_layout()
adjust_text(text)

fig.savefig('C:/Users/conwayjp/OneDrive - NIWA/projects/MarsdenFS2018/Obj1/Obs data/data_sites_RGI_Sept2021_shortnames.png', dpi=600)
plt.clf()
