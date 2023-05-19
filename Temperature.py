import sys
import time as tm
import netCDF4
# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from mpi4py import MPI
# from multiprocessing import Pool


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
        out3_local[lat_dict_local[lat_cons_local[k_local]], lon_dict_local[lon_cons_local[k_local]]] = out2_local[
            k_local]

    out4_local = griddata((lat2_local[~np.isnan(out3_local)], lon2_local[~np.isnan(out3_local)]),
                          out3_local[~np.isnan(out3_local)], (lat_cons_local, lon_cons_local), method='nearest')
    out4_local = np.array(out4_local)

    out5_local = np.zeros((len(lat2_local[:, 0]), len(lon2_local[0, :])))
    out5_local[:] = np.nan

    for k_local in np.arange(0, len(lon_cons_local)):
        out5_local[lat_dict_local[lat_cons_local[k_local]], lon_dict_local[lon_cons_local[k_local]]] = out4_local[
            k_local]

    return out5_local


def interpolate_sigma(arr):
    lat2_local, lon2_local, out4_local, bottomT2_local, depth_local, h_local, s_rho_local = arr

    out_final_local = np.zeros((len(s_rho_local), len(lat2_local[:, 0]), len(lon2_local[0, :])))
    out_final_local[:] = np.nan

    for i in np.arange(0, len(lat2_local[:, 0])):
        for j_local in np.arange(0, len(lon2_local[0, :])):
            z_local = np.array(out4_local[:, i, j_local])
            z_local = z_local[~np.isnan(z_local)]

            if len(z_local) == 0:
                continue

            z_local = np.resize(z_local, len(z_local) + 1)
            z_local[len(z_local) - 1] = bottomT2_local[i, j_local]
            depth_act = depth_local[0:len(z_local)]
            depth_act[len(z_local) - 1] = h_local[i, j_local]

            depth2 = np.abs(s_rho_local * h_local[i, j_local])

            out_final_local[:, i, j_local] = np.interp(depth2, depth_act, z_local)

    return out_final_local


# lat 37.27 42.22 lon 8.5 16.75

# lat 39.8 41.35 lon 13.1 15.85

# depth time lat lon thetao bottomT

if len(sys.argv) != 6:
    print("Usage: python " + str(sys.argv[0]) + "source_filename mask_filename destination_filename "
                                                "border_filename time")
    sys.exit(-1)

src_filename = sys.argv[1]
mask_filename = sys.argv[2]
destination_filename = sys.argv[3]
border_filename = sys.argv[4]
time = int(sys.argv[5])

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# source values
nc = xr.open_dataset(src_filename)
temp = nc.variables['thetao'][:]
temp = np.array(temp[time, :, :, :])
bottomT = nc.variables['bottomT'][:]
bottomT = bottomT[time, :, :]
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
nc.close()

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
nc_grid.close()

nc_mask = Dataset(mask_filename, "r+")
mask = nc_mask.variables['mask_rho'][:]
mask = np.array(mask)
nc_mask.close()

# lon2[0, :] lat2[:, 0]

# use the coordinate as key and index as value
lon_dict = {lon2[0, j]: j for j in np.arange(0, len(lon2[0, :]))}
lat_dict = {lat2[j, 0]: j for j in np.arange(0, len(lat2[:, 0]))}

last = len(depth)
if rank == 0:
    start = tm.time()
    # interpolate temperature on longitude and latitude
if rank == 0:
    start_x = tm.time()

    out2d = np.zeros((len(depth), len(lat2[:, 0]), len(lon2[0, :])))
    out2d[:] = np.nan

depth_rank = np.zeros((np.ceil(len(depth) / size)))
depth_rank[:] = np.nan
data = [temp, latf, lonf, lat2, lon2, depth, h, mask, lat_dict, lon_dict]
comm.Scatter(depth, depth_rank, root=0)
depth_rank = depth_rank[~np.isnan(depth_rank)]
out2d_rank = np.zeros((len(depth_rank), len(lat2[:, 0]), len(lon2[0, :])))
out2d_rank[:] = np.nan

for i in np.arange(0, len(depth_rank)):
    out2d_rank = interpolation_lat_lon(data, i)

comm.barrier()
comm.Gather(out2d_rank, out2d, root=0)

'''
items = [(data, i) for i in np.arange(0, len(depth))]
with Pool(processes=20) as p:
    result = p.starmap(interpolation_lat_lon, items)
'''

if rank == 0:

    # find the last index at witch we have data and move data to out2d
    for i in np.arange(0, len(depth)):
        if np.isnan(out2d[i]).size == 0 and i < last:
            last = i

    out4 = out2d[0:last, :, :]

    print("2d interpolation time:", tm.time() - start_x)

    # 875 secondi

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
    # 27 secondi
    # 28 secondi

    print("total time:", tm.time() - start)
    # 247 secondi
    # 936 secondi di esecuzione

    out_final[np.isnan(out_final)] = 1.e+37

    nc_destination = Dataset(destination_filename, "a")
    nc_destination.variables['temp'][time, :, :, :] = out_final[:]

    nc = Dataset(src_filename, "r+")
    time_var = nc.variables['time']
    dtime = netCDF4.num2date(time_var[:], time_var.units)
    desttime = netCDF4.date2num(dtime, nc_destination.variables['ocean_time'].units)
    desttime = desttime.astype(int)
    nc_destination.variables['ocean_time'][:] = desttime[:]
    nc.close()
    nc_destination.close()

    nc_border = Dataset(border_filename, "a")
    nc_border.variables['temp_west'][time, :, :] = out_final[0, :, 0]
    nc_border.variables['temp_south'][time, :, :] = out_final[0, 0, :]
    nc_border.variables['temp_east'][time, :, :] = out_final[0, :, -1]
    nc_border.variables['temp_north'][time, :, :] = out_final[0, -1, :]

    nc_border.variables['ocean_time'][:] = desttime[:]

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
    tem = map.contourf(x, y, out_final[0])
    cb = map.colorbar(tem, "right")

    plt.show()
'''

"""
2d interpolation time: 869.9637989997864 with  1  processes
2d interpolation time: 449.54563093185425 with  2  processes
2d interpolation time: 312.03366327285767 with  3  processes
2d interpolation time: 249.49427700042725 with  4  processes
2d interpolation time: 218.5005121231079 with  5  processes
2d interpolation time: 205.65114283561707 with  6  processes
2d interpolation time: 204.57789087295532 with  7  processes
2d interpolation time: 221.00535082817078 with  8  processes
2d interpolation time: 282.692747592926 with  9  processes
2d interpolation time: 333.8800582885742 with  10  processes
2d interpolation time: 359.7983169555664 with  11  processes
2d interpolation time: 504.4210469722748 with  12  processes
bottom temperature interpolation time: 13.32588791847229
sigma interpolation time: 47.563385009765625 with  1  processes
sigma interpolation time: 44.46232509613037 with  2  processes
sigma interpolation time: 61.36935782432556 with  3  processes
sigma interpolation time: 80.69376683235168 with  4  processes
sigma interpolation time: 100.81394720077515 with  5  processes
sigma interpolation time: 120.50593185424805 with  6  processes
sigma interpolation time: 141.50088691711426 with  7  processes
sigma interpolation time: 161.47735404968262 with  8  processes
"""
