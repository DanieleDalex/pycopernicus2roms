import sys
import time as tm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from scipy.interpolate import griddata
from multiprocessing import Pool


# file to test the best number of processes for optimizing the code

def interpolation_lat_lon(arr, i_local):
    temp_local, latf_local, lonf_local, lat2_local, lon2_local, depth_local, h_local, mask_local, \
        lat_dict_local, lon_dict_local = arr

    z_local = np.array(temp_local[i_local, :, :]).flatten()
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
    lat2_local, lon2_local, out4_local, bottomT2_local, depth_local, h_local, s_rho_local = arr

    out_final_local = np.zeros((len(s_rho_local), len(lat2_local[:, 0]), len(lon2_local[0, :])))
    out_final_local[:] = np.nan

    for i in np.arange(0, len(lat2_local[:, 0])):
        for j in np.arange(0, len(lon2_local[0, :])):
            z_local = np.array(out4_local[:, i, j])
            z_local = z_local[~np.isnan(z_local)]

            z_local = np.resize(z_local, len(z_local) + 1)
            z_local[len(z_local) - 1] = bottomT2_local[i, j]
            depth_act = depth_local[0:len(z_local)]
            depth_act[len(z_local) - 1] = h_local[i, j]

            depth2 = (s_rho_local * h_local[i, j]) * -1

            out_final_local[:, i, j] = np.interp(depth2, depth_act, z_local)

    return out_final_local


# lat 37.27 42.22 lon 8.5 16.75

# lat 39.8 41.35 lon 13.1 15.85

# depth time lat lon thetao bottomT
if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python " + str(sys.argv[0]) + " source_filename grid_filename")
        sys.exit(-1)

    src_filename = sys.argv[1]
    grid_filename = sys.argv[2]

    # source values
    nc = xr.open_dataset(src_filename)
    temp = nc.variables['thetao'][:]
    temp = np.array(temp[0, :, :, :])
    bottomT = nc.variables['bottomT'][:]
    bottomT = bottomT[0, :, :]
    bottomT = np.array(bottomT)

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
    nc_grid = Dataset(grid_filename, "r+", format="NETCDF4_CLASSIC")
    lon2 = nc_grid.variables['lon_rho'][:]
    lat2 = nc_grid.variables['lat_rho'][:]
    # depth2 = nc_grid.variables['s_rho'][:]
    h = nc_grid.variables['h'][:]
    mask = nc_grid.variables['mask_rho'][:]
    s_rho = [-0.983333333333333, -0.95, -0.916666666666667, -0.883333333333333, -0.85, -0.816666666666667,
             -0.783333333333333, -0.75, -0.716666666666667, -0.683333333333333, -0.65, -0.616666666666667,
             -0.583333333333333, -0.55, -0.516666666666667, -0.483333333333333, -0.45, -0.416666666666667,
             -0.383333333333333, -0.35, -0.316666666666667, -0.283333333333333, -0.25, -0.216666666666667,
             -0.183333333333333, -0.15, -0.116666666666667, -0.0833333333333333, -0.05, -0.0166666666666667]
    s_rho = np.array(s_rho)
    mask = np.array(mask)
    lon2 = np.array(lon2)
    lat2 = np.array(lat2)
    h = np.array(h)

    # lon2[0, :] lat2[:, 0]

    # use the coordinate as key and index as value
    lon_dict = {lon2[0, j]: j for j in np.arange(0, len(lon2[0, :]))}
    lat_dict = {lat2[j, 0]: j for j in np.arange(0, len(lat2[:, 0]))}

    last = len(depth)
    start = tm.time()
    # interpolate temperature on longitude and latitude
    for i in np.arange(1, 13):
        start_x = tm.time()

        out2d = np.zeros((len(depth), len(lat2[:, 0]), len(lon2[0, :])))
        out2d[:] = np.nan

        data = [temp, latf, lonf, lat2, lon2, depth, h, mask, lat_dict, lon_dict]
        items = [(data, i) for i in np.arange(0, len(depth))]
        with Pool(processes=i) as p:
            result = p.starmap(interpolation_lat_lon, items)

        print("2d interpolation time:", tm.time() - start_x, "with ", i, " processes")

    # find the last index at witch we have data and move data to out2d
    for i in np.arange(0, len(depth)):
        out2d[i, :, :] = result[i]
        if np.isnan(out2d[i]).size == 0 and i < last:
            last = i

    out4 = out2d[0:last, :, :]

    # interpolate temperature at sea floor
    start_b = tm.time()

    bottomT2 = np.zeros((len(lat2[:, 0]), len(lon2[0, :])))
    bottomT2[:] = np.nan

    z = np.array(bottomT).flatten()
    out = griddata((latf, lonf), z, (lat2, lon2), method='linear')

    lat_cons = lat2[mask == 1]
    lon_cons = lon2[mask == 1]
    lat_cons = np.array(lat_cons)
    lon_cons = np.array(lon_cons)

    out2 = griddata((lat2[~np.isnan(out)], lon2[~np.isnan(out)]), out[~np.isnan(out)],
                    (lat_cons, lon_cons), method='linear')

    out2 = np.array(out2)

    for k in np.arange(0, len(lon_cons)):
        bottomT2[lat_dict[lat_cons[k]], lon_dict[lon_cons[k]]] = out2[k]

    print("bottom temperature interpolation time:", tm.time() - start_b)
    # 13 secondi

    # interpolate temperature on sigma

    start_s = tm.time()

    data = [lat2, lon2, out4, bottomT2, depth, h, s_rho]

    out_final = interpolate_sigma(data)

    print("sigma interpolation time:", tm.time() - start_s)

    print("total time:", tm.time() - start)
