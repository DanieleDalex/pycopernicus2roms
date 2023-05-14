import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

nc_mine = Dataset("ini-d03.nc")
lon = nc_mine.variables['lon_rho'][:]
lon = np.array(lon)
lat = nc_mine.variables['lat_rho'][:]
lat = np.array(lat)

u_mine = nc_mine.variables['u'][:]
u_mine = np.array(u_mine)
v_mine = nc_mine.variables['v'][:]
v_mine = np.array(v_mine)
ubar_mine = nc_mine.variables['ubar'][:]
ubar_mine = np.array(ubar_mine)
vbar_mine = nc_mine.variables['vbar'][:]
vbar_mine = np.array(vbar_mine)
nc_mine.close()

nc_legacy = Dataset("ini-d03.nc.legacy")
u_legacy = nc_legacy.variables['u'][:]
u_legacy = np.array(u_legacy)
v_legacy = nc_legacy.variables['v'][:]
v_legacy = np.array(v_legacy)
ubar_legacy = nc_legacy.variables['ubar'][:]
ubar_legacy = np.array(ubar_legacy)
vbar_legacy = nc_legacy.variables['vbar'][:]
vbar_legacy = np.array(vbar_legacy)
nc_legacy.close()

u_diff = u_mine - u_legacy
v_diff = v_mine - v_legacy
ubar_diff = ubar_mine - ubar_legacy
vbar_diff = vbar_mine - vbar_legacy

map = Basemap(projection='merc', llcrnrlon=13., llcrnrlat=39.5, urcrnrlon=16., urcrnrlat=41.5,
                  resolution='i', ellps='WGS84')

parallels = np.arange(39, 42, 0.5)
meridians = np.arange(12, 17, 0.5)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

x, y = map(lon, lat)
# curr = map.contourf(x, y, u_diff[0, 0, :, :])
# cb = map.colorbar(curr, "right")
# quiver = map.quiver(x[::20, ::20], y[::20, ::20], u_diff[0, 0, ::20, ::20], v_diff[0, 0, ::20, ::20])

plt.show()

