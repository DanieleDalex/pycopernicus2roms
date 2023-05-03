import sys
import time as tm
# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from scipy.interpolate import griddata
from multiprocessing import Pool
# from ray.util.multiprocessing import Pool


def interpolation_lat_lon(arr, i_local):
    so_local, latf_local, lonf_local, lat2_local, lon2_local, depth_local, h_local, mask_local, \
        lat_dict_local, lon_dict_local = arr

    z_local = np.array(so_local[i_local, :, :]).flatten()
    out_local = griddata((latf_local, lonf_local), z_local, (lat2_local, lon2_local), method='linear')

    if out_local[~np.isnan(out_local)].size == 0:
        return np.nan

    depth_local = depth_local[i_local]

    if depth_local < 5:
        depth_local = 5

    lat_cons_local = lat2_local[np.logical_and(h_local >= depth_local, mask_local)]
    lon_cons_local = lon2_local[np.logical_and(h_local >= depth_local, mask_local)]
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


def interpolate_sigma(arr):
    lat2_local, lon2_local, out4_local, depth_local, h_local, s_rho_local = arr

    out_final_local = np.zeros((len(s_rho_local), len(lat2_local[:, 0]), len(lon2_local[0, :])))
    out_final_local[:] = np.nan

    for i in np.arange(0, len(lat2_local[:, 0])):
        for j in np.arange(0, len(lon2_local[0, :])):
            z_local = np.array(out4_local[:, i, j])
            z_local = z_local[~np.isnan(z_local)]

            if len(z_local) == 0:
                continue

            depth_act = depth_local[0:len(z_local)]

            depth2 = (s_rho_local * h_local[i, j]) * -1

            out_final_local[:, i, j] = np.interp(depth2, depth_act, z_local)

    return out_final_local


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
    so = nc.variables['so'][:]
    so = np.array(so[time, :, :, :])

    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]
    depth = nc.variables['depth'][:]
    depth = np.array(depth)

    nc_frame = nc.to_dataframe()
    nc_frame = nc_frame.reset_index()
    nc_frame = nc_frame.loc[nc_frame['time'] == nc_frame['time'][0]]
    nc_framel = nc_frame.loc[nc_frame['depth'] == nc_frame['depth'][0]]
    lonf = nc_framel['lon']
    latf = nc_framel['lat']
    lonf = np.array(lonf)
    latf = np.array(latf)

    # destination grid
    nc_grid = Dataset(destination_filename, "r+", format="NETCDF4_CLASSIC")
    lon2 = nc_grid.variables['lon_rho'][:]
    lat2 = nc_grid.variables['lat_rho'][:]
    lon2 = np.array(lon2)
    lat2 = np.array(lat2)

    h = nc_grid.variables['h'][:]
    h = np.array(h)

    s_rho = nc_grid.variables['s_rho'][:]
    s_rho = np.array(s_rho)

    nc_mask = Dataset(mask_filename, "r+")
    mask = nc_mask.variables['mask_rho'][:]
    mask = np.array(mask)
    nc_mask.close()

    # use the coordinate as key and index as value
    lon_dict = {lon2[0, j]: j for j in np.arange(0, len(lon2[0, :]))}
    lat_dict = {lat2[j, 0]: j for j in np.arange(0, len(lat2[:, 0]))}

    last = len(depth)
    start = tm.time()

    # interpolate salinity on longitude and latitude

    start_x = tm.time()

    out2d = np.zeros((len(depth), len(lat2[:, 0]), len(lon2[0, :])))
    out2d[:] = np.nan

    data = [so, latf, lonf, lat2, lon2, depth, h, mask, lat_dict, lon_dict]
    items = [(data, i) for i in np.arange(0, len(depth))]
    with Pool(processes=20) as p:
        result = p.starmap(interpolation_lat_lon, items)

    print("2d interpolation time:", tm.time() - start_x, "with ", 6, " processes")

    # find the last index at witch we have data and move data to out2d
    for i in np.arange(0, len(depth)):
        out2d[i, :, :] = result[i]
        if np.isnan(out2d[i]).size == 0 and i < last:
            last = i

    out4 = out2d[0:last, :, :]

    # interpolate temperature on sigma

    start_s = tm.time()

    data = [lat2, lon2, out4, depth, h, s_rho]

    out_final = interpolate_sigma(data)

    print("sigma interpolation time:", tm.time() - start_s)

    print("total time:", tm.time() - start)

    nc_destination = Dataset(destination_filename, "a")
    nc_destination.variables['salt'][time, :, :, :] = out_final[:]
    nc_destination.close()

    nc_border = Dataset(border_filename, "a")
    nc_border.variables['salt_west'][time, :, :] = out_final[0, :, 0]
    nc_border.variables['salt_south'][time, :, :] = out_final[0, 0, :]
    nc_border.variables['salt_east'][time, :, :] = out_final[0, :, -1]
    nc_border.variables['salt_nord'][time, :, :] = out_final[0, -1, :]
    nc_border.close()

    '''
    for k in np.arange(0, len(s_rho)):
        plt.figure(k)

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
        tem = map.contourf(x, y, out_final[k])
        cb = map.colorbar(tem, "right")

        plt.show()
    '''
