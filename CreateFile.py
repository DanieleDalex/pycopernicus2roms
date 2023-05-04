import sys

from netCDF4 import Dataset


if len(sys.argv) != 5:
    print("Usage: python " + str(sys.argv[0]) + " grid_filename destination_file grid_border_file border_file")
    sys.exit(-1)

grid_filename = sys.argv[1]
dst = sys.argv[2]
grid_border = sys.argv[3]
border = sys.argv[4]

# Open the NetCDF domain grid file
ncgridfile = Dataset(grid_filename)

# Create an empty destination file
ncdstfile = Dataset(dst, "w", format="NETCDF4")

# set attributes
ncdstfile.setncatts(ncgridfile.__dict__)

# Create dimensions
for name, dimension in ncgridfile.dimensions.items():
    ncdstfile.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

# create variables
for name, variable in ncgridfile.variables.items():
    x = ncdstfile.createVariable(name, variable.datatype, variable.dimensions)
    ncdstfile[name].setncatts(ncgridfile[name].__dict__)

# open grid border file
ncgridborder = Dataset(grid_border, "r+")

# create empty grid file
ncborder = Dataset(border, "w", format="NETCDF4")

# set attributes
ncborder.setncatts(ncgridborder.__dict__)

# create dimensions
for name, dimension in ncgridborder.dimensions.items():
    ncborder.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

# create variables
for name, variable in ncgridborder.variables.items():
    x = ncborder.createVariable(name, variable.datatype, variable.dimensions)
    ncborder[name].setncatts(ncgridborder[name].__dict__)


'''
# Create variables
lat = ncdstfile.createVariable("lat", "f8", ("eta_rho", "xi_rho"))
lat.long_name = "latitude of RHO-points"
lat.units = "degree_north"
lat.field = "lat_rho, scalar"
lat.standard_name = "latitude"
lat._CoordinateAxisType = "Lat"

lon = ncdstfile.createVariable("lon", "f8", ("eta_rho", "xi_rho"))
lon.long_name = "longitude of RHO-points"
lon.units = "degree_east"
lon.field = "lon_rho, scalar"
lon.standard_name = "longitude"
lon._CoordinateAxisType = "Lon"

time = ncdstfile.createVariable("time", "f8", "ocean_time")
time.long_name = "atmospheric forcing time"
time.units = "days since 1968-05-23 00:00:00 GMT"
time.calendar = "gregorian"

Pair = ncdstfile.createVariable("Pair", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
Pair.long_name = "Mean Sea Level Pressure"
Pair.units = "millibar"
Pair.time = "time"

Tair = ncdstfile.createVariable("Tair", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
Tair.long_name = "Air Temperature (2m)"
Tair.units = "Celsius"
Tair.time = "time"

Qair = ncdstfile.createVariable("Qair", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
Qair.long_name = "Relative Humidity (2m)"
Qair.units = "percentage"
Qair.time = "time"

rain = ncdstfile.createVariable("rain", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
rain.long_name = "Rain fall rate"
rain.units = "kilogram meter-2 second-1"
rain.time = "time"

swrad = ncdstfile.createVariable("swrad", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
swrad.long_name = "Solar showtwave radiation"
swrad.units = "watt meter-2"
swrad.time = "time"
swrad.positive_value = "downward flux, heating"
swrad.negative_value = "upward flux, cooling"

lwrad_down = ncdstfile.createVariable("lwrad_down", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
lwrad_down.long_name = "Net longwave radiation flux"
lwrad_down.units = "watt meter-2"
lwrad_down.time = "time"
lwrad_down.positive_value = "downward flux, heating"
lwrad_down.negative_value = "upward flux, cooling"

Uwind = ncdstfile.createVariable("Uwind", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
Uwind.long_name = "Wind velocity, u-component (m s-1)"
Uwind.description = "grid rel. x-wind component"
Uwind.units = "m s-1"
Uwind.time = "time"

Vwind = ncdstfile.createVariable("Vwind", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.e+37)
Vwind.long_name = "Wind velocity, v-component (m s-1)"
Vwind.description = "grid rel. y-wind component"
Vwind.units = "m s-1"
Vwind.time = "time"

lat_u = ncdstfile.createVariable("lat_u", "f8", ("eta_u", "xi_u"))
lat_u.long_name = "latitude of U-points"
lat_u.units = "degree_north"
lat_u.standard_name = "latitude"
lat_u._CoordinateAxisType = "Lat"

lon_u = ncdstfile.createVariable("lon_u", "f8", ("eta_u", "xi_u"))
lon_u.long_name = "longitude of U_points"
lon_u.units = "degree_east"
lon_u.standard_name = "longitude"
lon_u._CoordinateAxisType = "Lon"

lat_v = ncdstfile.createVariable("lat_v", "f8", ("eta_v", "xi_v"))
lat_v.long_name = "latitude of V-points"
lat_v.units = "degree_north"
lat_v.standard_name = "latitude"
lat_v._CoordinateAxisType = "Lat"

lon_v = ncdstfile.createVariable("lon_v", "f8", ("eta_v", "xi_v"))
lon_v.long_name = "longitude of V-points"
lon_v.units = "degree_east"
lon_v.standard_name = "longitude"
lon_v._CoordinateAxisType = "Lon"

ocean_time_var = ncdstfile.createVariable("ocean_time", "f8", "ocean_time")
ocean_time_var.long_name = "surface ocean time"
ocean_time_var.units = "days since 1968-05-23 00:00:00 GMT"
ocean_time_var.calendar = "gregorian"

sustr = ncdstfile.createVariable("sustr", "f4", ("ocean_time", "eta_u", "xi_u"), fill_value=1.e+37)
sustr.long_name = "Kinematic wind stress, u-component (m2 s-2)"
sustr.units = "Newton meter-2"
sustr.scale_factor = 1000.
sustr.time = "ocean_time"

svstr = ncdstfile.createVariable("svstr", "f4", ("ocean_time", "eta_v", "xi_v"), fill_value=1.e+37)
svstr.long_name = "Kinematic wind stress, v-component (m2 s-2)"
svstr.units = "Newton meter-2"
svstr.scale_factor = 1000.
svstr.time = "ocean_time"

u = ncdstfile.createVariable("u", "f4", ("ocean_time", "s_rho", "eta_u", "xi_u"), fill_value=1.e+37)
u.long_name = "u-momentum component"
u.units = "meter second-1"
u.coordinates = "lon_u lat_u sc_r ocean_time"
u.time = "ocean_time"

v = ncdstfile.createVariable("v", "f4", ("ocean_time", "s_rho", "eta_v", "xi_v"), fill_value=1.e+37)
v.long_name = "v-momentum component"
v.units = "meter second-1"
v.coordinates = "lon_v lat_v sc_r ocean_time"
v.time = "ocean_time"
'''

# assign longitude and latitude from grid to destination
ncdstfile.variables['lat_rho'][:] = ncgridfile.variables['lat_rho'][:]
ncdstfile.variables['lon_rho'][:] = ncgridfile.variables['lon_rho'][:]

ncdstfile.variables['lat_u'][:] = ncgridfile.variables['lat_u'][:]
ncdstfile.variables['lon_u'][:] = ncgridfile.variables['lon_u'][:]

ncdstfile.variables['lat_v'][:] = ncgridfile.variables['lat_v'][:]
ncdstfile.variables['lon_v'][:] = ncgridfile.variables['lon_v'][:]

ncdstfile.variables['s_rho'][:] = ncgridfile.variables['s_rho'][:]

ncdstfile.variables['h'][:] = ncgridfile.variables['h'][:]

ncdstfile.variables['theta_b'][:] = ncgridfile.variables['theta_b'][:]
ncdstfile.variables['theta_s'][:] = ncgridfile.variables['theta_s'][:]

ncdstfile.variables['hc'][:] = ncgridfile.variables['hc'][:]

ncdstfile.varialbes['Tcline'][:] = ncgridfile.variables['Tcline'][:]

ncdstfile.variables['tend'][:] = ncgridfile.variables['tend'][:]

ncdstfile.variables['scrum_time'][:] = ncgridfile.variables['scrum_time'][:]

ncdstfile.variables['Cs_r'][:] = ncgridfile.variables['Cs_r'][:]
ncdstfile.variables['sc_r'][:] = ncgridfile.variables['sc_r'][:]
ncdstfile.close()


# assign longitude and latitude from grid to border
ncborder.variables['lat_rho'][:] = ncgridborder.variables['lat_rho'][:]
ncborder.variables['lon_rho'][:] = ncgridborder.variables['lon_rho'][:]

ncborder.variables['lat_u'][:] = ncgridborder.variables['lat_u'][:]
ncborder.variables['lon_u'][:] = ncgridborder.variables['lon_u'][:]

ncborder.variables['lat_v'][:] = ncgridborder.variables['lat_v'][:]
ncborder.variables['lon_v'][:] = ncgridborder.variables['lon_v'][:]

ncborder.variables['s_rho'][:] = ncgridborder.variables['s_rho'][:]

ncborder.variables['h'][:] = ncgridborder.variables['h'][:]

ncborder.variables['theta_b'][:] = ncgridborder.variables['theta_b'][:]
ncborder.variables['theta_s'][:] = ncgridborder.variables['theta_s'][:]

ncborder.variables['hc'][:] = ncgridborder.variables['hc'][:]

ncborder.variables['Tcline'][:] = ncgridborder.variables['Tcline'][:]

ncborder.variables['z_r'][:] = ncgridborder.variables['z_r'][:]

ncborder.variables['Cs_r'][:] = ncgridborder.variables['Cs_r'][:]
ncborder.variables['sc_r'][:] = ncgridborder.variables['sc_r'][:]
ncborder.close()
