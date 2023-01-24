import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import matplotlib.pyplot as plt

# lat 37.27 42.22 lon 8.5 16.75

# lat 39.8 41.35 lon 13.1 15.85

# depth time lat lon thetao bottomT

if len(sys.argv) != 3:
    print("Usage: python " + str(sys.argv[0]) + " source_filename grid_filename")
    sys.exit(-1)

src_filename = sys.argv[1]
grid_filename = sys.argv[2]

nc = xr.open_dataset(src_filename)
temp = nc.variables['thetao'][:]
temp = temp[0, 0, :, :]
nc = nc.to_dataframe()
nc = nc.reset_index()
nc = nc.loc[nc['depth'] == nc['depth'][0]]
nc = nc.loc[nc['time'] == nc['time'][0]]
lon = nc['lon']
lat = nc['lat']


nc2 = Dataset(grid_filename, "r+", format="NETCDF4_CLASSIC")
lon2 = nc2.variables['lon_rho'][:]
lat2 = nc2.variables['lat_rho'][:]

px = np.array(lon).flatten()
py = np.array(lat).flatten()

z = np.array(temp).flatten()
z[z == 1.e+37] = 'nan'

X, Y = np.meshgrid(lon2, lat2, sparse=True)
out = griddata((px, py), z, (X, Y), method='linear', fill_value=1.e+37)
print(out.shape)

# print(nc.variables)


# plt.show()
