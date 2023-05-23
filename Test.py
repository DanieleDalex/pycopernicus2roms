import sys
import time as tm
# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from multiprocessing import Pool
from scipy.interpolate import SmoothBivariateSpline


def interpolation_lat_lon(arr, i_local, kxy_local):
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

    spline = SmoothBivariateSpline(lat2_local[~np.isnan(out_local)], lon2_local[~np.isnan(out_local)],
                                   out_local[~np.isnan(out_local)], kx=kxy_local, ky=kxy_local)

    out2_local = spline(lat_cons_local, lon_cons_local, grid=False)

    out3_local = np.zeros((len(lat2_local[:, 0]), len(lon2_local[0, :])))
    out3_local[:] = np.nan

    for k_local in np.arange(0, len(lon_cons_local)):
        out3_local[lat_dict_local[lat_cons_local[k_local]], lon_dict_local[lon_cons_local[k_local]]] = out2_local[
            k_local]

    return out3_local


def interpolate_sigma(arr):
    lat2_local, lon2_local, out4_local, depth_local, h_local, s_rho_local = arr

    out_final_local = np.zeros((len(s_rho_local), len(lat2_local[:, 0]), len(lon2_local[0, :])))
    out_final_local[:] = np.nan

    for i in np.arange(0, len(lat2_local[:, 0])):
        for j_local in np.arange(0, len(lon2_local[0, :])):
            z_local = np.array(out4_local[:, i, j_local])
            z_local = z_local[~np.isnan(z_local)]

            if len(z_local) == 0:
                continue

            if len(z_local) == 1:
                out_final_local[:, i, j_local] = z_local
                continue

            depth_act = depth_local[0:len(z_local)]

            depth2 = np.abs(s_rho_local * h_local[i, j_local])

            out_final_local[:, i, j_local] = np.interp(depth2, depth_act, z_local)

    return out_final_local


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print("Usage: python " + str(sys.argv[0]) + "source_filename mask_filename destination_filename "
                                                    "time legacy")
        sys.exit(-1)

    src_filename = sys.argv[1]
    mask_filename = sys.argv[2]
    destination_filename = sys.argv[3]
    time = int(sys.argv[4])
    legacy_filename = sys.argv[5]

    # source values
    nc = xr.open_dataset(src_filename)
    uo = nc.variables['uo'][:]
    uo = np.array(uo[time, :, :, :])
    vo = nc.variables['vo'][:]
    vo = np.array(vo[time, :, :, :])

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
    lon2_u = nc_grid.variables['lon_u'][:]
    lat2_u = nc_grid.variables['lat_u'][:]
    lon2_u = np.array(lon2_u)
    lat2_u = np.array(lat2_u)
    lon2_v = nc_grid.variables['lon_v'][:]
    lat2_v = nc_grid.variables['lat_v'][:]
    lon2_v = np.array(lon2_v)
    lat2_v = np.array(lat2_v)

    h = nc_grid.variables['h'][:]
    h = np.array(h)

    s_rho = nc_grid.variables['s_rho'][:]
    s_rho = np.array(s_rho)
    nc_grid.close()

    nc_mask = Dataset(mask_filename, "r+")
    mask = nc_mask.variables['mask_rho'][:]
    mask_u = nc_mask.variables['mask_u'][:]
    mask_v = nc_mask.variables['mask_v'][:]
    mask = np.array(mask)
    mask_u = np.array(mask_u)
    mask_v = np.array(mask_v)

    '''
    s_rho = [-0.983333333333333, -0.95, -0.916666666666667, -0.883333333333333, -0.85, -0.816666666666667,
             -0.783333333333333, -0.75, -0.716666666666667, -0.683333333333333, -0.65, -0.616666666666667,
             -0.583333333333333, -0.55, -0.516666666666667, -0.483333333333333, -0.45, -0.416666666666667,
             -0.383333333333333, -0.35, -0.316666666666667, -0.283333333333333, -0.25, -0.216666666666667,
             -0.183333333333333, -0.15, -0.116666666666667, -0.0833333333333333, -0.05, -0.0166666666666667]
    s_rho = np.array(s_rho)
    '''

    angle = nc_mask.variables['angle'][:]

    nc_legacy = Dataset(legacy_filename, "r+")

    u_legacy = nc_legacy.variables['u'][:]
    u_legacy[u_legacy == 1.e+37] = np.nan
    # lon2[0, :] lat2[:, 0]

    # use the coordinate as key and index as value

    lon_dict = {lon2[0, j]: j for j in np.arange(0, len(lon2[0, :]))}
    lat_dict = {lat2[j, 0]: j for j in np.arange(0, len(lat2[:, 0]))}

    lon_dict_u = {lon2_u[0, j]: j for j in np.arange(0, len(lon2_u[0, :]))}
    lat_dict_u = {lat2_u[j, 0]: j for j in np.arange(0, len(lat2_u[:, 0]))}

    lon_dict_v = {lon2_v[0, j]: j for j in np.arange(0, len(lon2_v[0, :]))}
    lat_dict_v = {lat2_v[j, 0]: j for j in np.arange(0, len(lat2_v[:, 0]))}

    last_u = len(depth)
    last_v = len(depth)
    start = tm.time()

    # interpolate h on u and v
    h_u = griddata((lat2.flatten(), lon2.flatten()), h.flatten(), (lat2_u, lon2_u), method='linear')
    h_u = np.reshape(h_u, (len(lat2_u[:, 0]), len(lon2_u[0, :])))

    h_v = griddata((lat2.flatten(), lon2.flatten()), h.flatten(), (lat2_v, lon2_v), method='linear')
    h_v = np.reshape(h_v, (len(lat2_v[:, 0]), len(lon2_v[0, :])))

    best = np.nan
    best_k = np.nan

    for k in np.arange(0, (len(lon2 * lat2)/2)):

        print("iniziamo la k numero: ", k)

        # interpolate current on u coordinate on longitude and latitude

        start_x = tm.time()

        out2d_u = np.zeros((len(depth), len(lat2_u[:, 0]), len(lon2_u[0, :])))
        out2d_u[:] = np.nan

        data = [uo, latf, lonf, lat2_u, lon2_u, depth, h_u, mask_u, lat_dict_u, lon_dict_u]
        items = [(data, i) for i in np.arange(0, len(depth))]
        with Pool(processes=20) as p:
            result = p.starmap(interpolation_lat_lon, items, k)

        # find the last index at witch we have data and move data to out2d
        for i in np.arange(0, len(depth)):
            out2d_u[i, :, :] = result[i]
            if np.isnan(out2d_u[i]).size == 0 and i < last_u:
                last_u = i

        out4_u = out2d_u[0:last_u, :, :]

        # interpolate current on v coordinate on longitude and latitude

        out2d_v = np.zeros((len(depth), len(lat2_v[:, 0]), len(lon2_v[0, :])))
        out2d_v[:] = np.nan

        data = [vo, latf, lonf, lat2_v, lon2_v, depth, h_v, mask_v, lat_dict_v, lon_dict_v]
        items = [(data, i) for i in np.arange(0, len(depth))]
        with Pool(processes=20) as p:
            result = p.starmap(interpolation_lat_lon, items, k)

        # find the last index at witch we have data and move data to out2d
        for i in np.arange(0, len(depth)):
            out2d_v[i, :, :] = result[i]
            if np.isnan(out2d_v[i]).size == 0 and i < last_v:
                last_v = i

        out4_v = out2d_v[0:last_v, :, :]

        print("2d interpolation time:", tm.time() - start_x)

        # interpolate current on sigma

        start_s = tm.time()

        data = [lat2_u, lon2_u, out4_u, depth, h_u, s_rho]

        out_final_u = interpolate_sigma(data)

        data = [lat2_v, lon2_v, out4_v, depth, h_v, s_rho]

        out_final_v = interpolate_sigma(data)

        print("sigma interpolation time:", tm.time() - start_s)

        print("total time:", tm.time() - start)

        u_diff = np.abs(np.abs(out_final_u[:]) - np.abs(u_legacy[:]))
        u_diff = u_diff[u_diff > 0.05]

        if np.isnan(best):
            best = len(u_diff)
            best_k = k

        if len(u_diff < best):
            best = len(u_diff)
            best_k = k

    print("best: ", best)
    print("best_k: ", best_k)

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
    x, y = map(lon2_u, lat2_u)
    # tem = map.contourf(x, y, out_final_u[:, :, 0])
    # cb = map.colorbar(tem, "right")
    # quiver = map.quiver(x[::20, ::20], y[::20, ::20], out_final_u[::20, ::20, 0], out_final_v[::20, ::20, 0])
    quiver = map.quiver(x[::20, ::20], y[::20, ::20], border_u[::20, ::20], border_v[::20, ::20])

    plt.show()
    '''
