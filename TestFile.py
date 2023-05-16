import sys
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

if len(sys.argv) != 3:
    print("Usage: python " + str(sys.argv[0]) + "mine_filename legacy_filename")
    sys.exit(-1)

mine_filename = sys.argv[1]
legacy_filename = sys.argv[2]

nc_mine = Dataset(mine_filename)
lon = nc_mine.variables['lon_rho'][:]
lon = np.array(lon)
lat = nc_mine.variables['lat_rho'][:]
lat = np.array(lat)
lon_u = nc_mine.variables['lon_u'][:]
lon_u = np.array(lon_u)
lat_u = nc_mine.variables['lat_u'][:]
lat_u = np.array(lat_u)
lon_v = nc_mine.variables['lon_v'][:]
lon_v = np.array(lon_v)
lat_v = nc_mine.variables['lat_v'][:]
lat_v = np.array(lat_v)

temp_mine = nc_mine.variables['temp'][:]
temp_mine = np.array(temp_mine)

'''
u_mine = nc_mine.variables['u'][:]
u_mine = np.array(u_mine)
v_mine = nc_mine.variables['v'][:]
v_mine = np.array(v_mine)
ubar_mine = nc_mine.variables['ubar'][:]
ubar_mine = np.array(ubar_mine)
vbar_mine = nc_mine.variables['vbar'][:]
vbar_mine = np.array(vbar_mine)
salt_mine = nc_mine.variables['salt'][:]
zeta_mine = nc_mine.variables['zeta'][:]
'''

nc_mine.close()

nc_legacy = Dataset(legacy_filename)
temp_legacy = nc_legacy.variables['temp'][:]
temp_legacy = np.array(temp_legacy)

'''
u_legacy = nc_legacy.variables['u'][:]
u_legacy = np.array(u_legacy)
v_legacy = nc_legacy.variables['v'][:]
v_legacy = np.array(v_legacy)
ubar_legacy = nc_legacy.variables['ubar'][:]
ubar_legacy = np.array(ubar_legacy)
vbar_legacy = nc_legacy.variables['vbar'][:]
vbar_legacy = np.array(vbar_legacy)
salt_legacy = nc_legacy.variables['salt'][:]
zeta_legacy = nc_legacy.variables['zeta'][:]
'''

nc_legacy.close()

temp_mine[temp_mine == 1.e+37] = np.nan

'''
u_mine[u_mine == 1.e+37] = np.nan
v_mine[v_mine == 1.e+37] = np.nan
ubar_mine[ubar_mine == 1.e+37] = np.nan
vbar_mine[vbar_mine == 1.e+37] = np.nan
salt_mine[salt_mine == 1.e+37] = np.nan
zeta_mine[zeta_mine == 1.e+37] = np.nan
'''

temp_legacy[temp_legacy == 1.e+37] = np.nan

'''
u_legacy[u_legacy == 1.e+37] = np.nan
v_legacy[v_legacy == 1.e+37] = np.nan
ubar_legacy[ubar_legacy == 1.e+37] = np.nan
vbar_legacy[vbar_legacy == 1.e+37] = np.nan
salt_legacy[salt_legacy == 1.e+37] = np.nan
zeta_legacy[zeta_legacy == 1.e+37] = np.nan
'''

temp_diff = temp_mine[:] - temp_legacy[:]

# u_diff = u_mine[:] - u_legacy[:]
# v_diff = v_mine[:] - v_legacy[:]
# ubar_diff = ubar_mine[:] - ubar_legacy[:]
# vbar_diff = vbar_mine[:] - vbar_legacy[:]

map = Basemap(projection='merc', llcrnrlon=13., llcrnrlat=39.5, urcrnrlon=16., urcrnrlat=41.5,
                  resolution='i', ellps='WGS84')

parallels = np.arange(39, 42, 0.5)
meridians = np.arange(12, 17, 0.5)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

x, y = map(lon, lat)
curr = map.contourf(x, y, temp_diff[0, 10, :, :])
cb = map.colorbar(curr, "right")
# quiver = map.quiver(x[::20, ::20], y[::20, ::20], u_diff[0, 0, ::20, ::20], v_diff[0, 0, ::20, ::20])

plt.show()

