import sys
import time as tm
# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from scipy.interpolate import griddata


def interpolation_lat_lon(arr):
    zos_local, latf_local, lonf_local, lat2_local, lon2_local, mask_local, \
        lat_dict_local, lon_dict_local = arr

    z_local = np.array(zos_local).flatten()
    out_local = griddata((latf_local, lonf_local), z_local, (lat2_local, lon2_local), method='linear')

    lat_cons_local = lat2_local[mask_local == 1]
    lon_cons_local = lon2_local[mask_local == 1]
    lat_cons_local = np.array(lat_cons_local)
    lon_cons_local = np.array(lon_cons_local)

    out2_local = griddata((lat2_local[~np.isnan(out_local)], lon2_local[~np.isnan(out_local)]),
                          out_local[~np.isnan(out_local)], (lat_cons_local, lon_cons_local), method='linear')

    out2_local = np.array(out2_local)

    out3_local = np.zeros((len(lat2_local[:, 0]), len(lon2_local[0, :])))
    out3_local[:] = np.nan

    for k_local in np.arange(0, len(lon_cons_local)):
        out3_local[lat_dict_local[lat_cons_local[k_local]], lon_dict_local[lon_cons_local[k_local]]] = out2_local[k_local]

    return out3_local


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print("Usage: python " + str(sys.argv[0]) + "source_filename mask_filename destination_filename "
                                                    "border_filename time")
        sys.exit(-1)

    src_filename = sys.argv[1]
    mask_filename = sys.argv[2]
    destination_filename = sys.argv[3]
    border_filename = sys.argv[4]
    time = int(sys.argv[5])

    # source values
    nc = xr.open_dataset(src_filename)
    zos = nc.variables['zos'][:]
    zos = np.array(zos[time, :, :])

    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]

    nc_frame = nc.to_dataframe()
    nc_frame = nc_frame.reset_index()
    nc_frame = nc_frame.loc[nc_frame['time'] == nc_frame['time'][0]]
    lonf = nc_frame['lon']
    latf = nc_frame['lat']
    lonf = np.array(lonf)
    latf = np.array(latf)

    # destination grid
    nc_grid = Dataset(destination_filename, "r+", format="NETCDF4_CLASSIC")

    lon2 = nc_grid.variables['lon_rho'][:]
    lat2 = nc_grid.variables['lat_rho'][:]
    lon2 = np.array(lon2)
    lat2 = np.array(lat2)

    nc_mask = Dataset(mask_filename, "r+")
    mask = nc_mask.variables['mask_rho'][:]
    mask = np.array(mask)
    nc_mask.close()

    # use the coordinate as key and index as value
    lon_dict = {lon2[0, j]: j for j in np.arange(0, len(lon2[0, :]))}
    lat_dict = {lat2[j, 0]: j for j in np.arange(0, len(lat2[:, 0]))}

    start = tm.time()

    # interpolate salinity on longitude and latitude

    start_x = tm.time()

    out2d = np.zeros((len(lat2[:, 0]), len(lon2[0, :])))
    out2d[:] = np.nan

    data = [zos, latf, lonf, lat2, lon2, mask, lat_dict, lon_dict]
    out2d = interpolation_lat_lon(data)

    print("2d interpolation time:", tm.time() - start_x)

    nc_destination = Dataset(destination_filename, "a")
    nc_destination.variables['zeta'][time, :, :] = out2d[:]
    nc_destination.close()

    nc_border = Dataset(border_filename, "a")
    nc_border.variables['zeta_west'][time, :] = out2d[:, 0]
    nc_border.variables['zeta_south'][time, :] = out2d[0, :]
    nc_border.variables['zeta_east'][time, :] = out2d[:, -1]
    nc_border.variables['zeta_nord'][time, :] = out2d[-1, :]
    nc_border.close()

    '''
    map = Basemap(projection='merc', llcrnrlon=13., llcrnrlat=39.5, urcrnrlon=16., urcrnrlat=41.5,
                      resolution='i', ellps='WGS84')

    parallels = np.arange(39, 42, 0.5)
    meridians = np.arange(12, 17, 0.5)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)
    # map.readshapefile("../ProvaNetCDF/Fwd_ Shapefile Campania/Campania_wgs84", 'Campania')

    # X, Y = np.meshgrid(lon, lat)
    # x, y = map(X, Y)
    x, y = map(lon2, lat2)
    tem = map.contourf(x, y, out2d)
    cb = map.colorbar(tem, "right")

    plt.show()
    '''
