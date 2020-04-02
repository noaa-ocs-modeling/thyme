# noinspection SpellCheckingInspection
"""
Utility classes and methods for working with HYCOM output.

The Hybrid Coordinate Ocean Model (HYCOM) is "a data-assimilative hybrid
isopycnal-sigma-pressure (generalized) coordinate ocean model." See
https://www.hycom.org for more information.

This module provides functionality allowing HYCOM output to be interpolated to
a regular, orthogonal lat/lon horizontal grid at a given depth-below-surface.

Currently, this module has only been tested to work with the HYCOM-based
Global Real-Time Ocean Forecast System (G-RTOFS) produced by NOAA's National
Centers for Environmental Prediction (NCEP), and would likely require
modifications to support other HYCOM-based model output.
"""
import netCDF4
import numpy
from osgeo import ogr, osr
from scipy import interpolate

from thyme.model import model
from thyme.util import interp

# Default fill value for NetCDF variables
FILLVALUE = -9999.0


class HYCOMIndexFile(model.ModelIndexFile):
    """NetCDF file containing metadata/grid info used during HYCOM processing.

    Store a regular grid definition, mask, and other information needed to
    process/convert native output from an HYCOM-based hydrodynamic model,
    within a reusable NetCDF file.

    Support is included for defining a set of regular, orthogonal subgrids that
    allow the data to be subset into multiple sub-domains during processing.
    This is accomplished by specifying a polygon shapefile containing one or
    more rectangular, orthogonal polygons defining areas where output data will
    be cropped and written to distinct output files.

    A unique model index file must be created for each combination of model,
    output grid resolution, land mask, and subset grid definition, and must be
    regenerated if anything changes (i.e., when a model domain extent is
    modified or the target output grid is redefined, a new model index file
    must be created before processing can resume). Until any of these
    properties change, the index file may be kept on the data processing system
    and reused in perpetuity.
    """

    def __init__(self, path):
        super().__init__(path)

    def compute_grid_mask(self, model_file, reg_grid):
        """Create model domain mask and write to index file.

        Args:
            model_file: `HYCOMOutputFile` instance containing irregular grid
                structure and variables.
            reg_grid: `RegularGrid` instance describing the regular grid for
                which the mask will be created.
        """
        # Create OGR layer in memory
        driver = ogr.GetDriverByName('Memory')
        dset = driver.CreateDataSource('grid_cell_mask')
        dset_srs = osr.SpatialReference()
        dset_srs.ImportFromEPSG(4326)
        layer = dset.CreateLayer('', dset_srs, ogr.wkbMultiPolygon)
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

        # Create polygons for each irregular grid cell
        # searching counter clockwise(eta1, xi), (eta2, xi2), (eta3, xi3),(eta4, xi4)
        for nx1 in range(model_file.num_x-1):
            for ny1 in range(model_file.num_y-1):
                nx2 = nx1 + 1
                ny2 = ny1
                nx3 = nx1 + 1
                ny3 = ny1 + 1
                nx4 = nx1
                ny4 = ny1 + 1

                # Search valid points
                valid_points = []
                for (ny, nx) in ((ny1, nx1), (ny2, nx2), (ny3, nx3), (ny4, nx4)):
                    if model_file.var_u.mask[0, 0, ny, nx]:
                        continue
                    valid_points.append((ny, nx))
                if len(valid_points) < 3:
                    continue
                ring = ogr.Geometry(ogr.wkbLinearRing)
                for (ny, nx) in valid_points:
                    ring.AddPoint(model_file.var_lon[ny, nx], model_file.var_lat[ny, nx])
                (ny, nx) = valid_points[0]
                ring.AddPoint(model_file.var_lon[ny, nx], model_file.var_lat[ny, nx])

                # Create polygon
                geom = ogr.Geometry(ogr.wkbPolygon)
                geom.AddGeometry(ring)
                feat = ogr.Feature(layer.GetLayerDefn())
                feat.SetField('id', 1)
                feat.SetGeometry(geom)
                layer.CreateFeature(feat)

        return self.rasterize_mask(reg_grid, layer)


class HYCOMFile(model.ModelFile):
    """Read/process data from a HYCOM model NetCDF file.

    Attributes:
        path: Path (relative or absolute) of the file.
    """

    def __init__(self, path, datetime_rounding=None):
        """Initialize HYCOM file object and open file at specified path.

        Args:
            path: Path of target NetCDF file.
            datetime_rounding: The `dateutil.DatetimeRounding` constant
                representing how date/time values should be rounded, or None if
                no rounding should occur.
        """
        super().__init__(path, datetime_rounding=datetime_rounding)
        self.var_lat = None
        self.var_lon = None
        self.var_u = None
        self.var_v = None
        self.var_date = None
        self.var_x = None
        self.var_y = None
        self.var_depth = None
        self.var_time = None
        self.time_units = None
        self.datetime_values = None
        self.var_mask = None
        self.num_x = None
        self.num_y = None
        self.num_times = None

    def release_resources(self):
        """Allow GC to reclaim memory by releasing/deleting resources."""
        self.var_lat = None
        self.var_lon = None
        self.var_u = None
        self.var_v = None
        self.var_date = None
        self.var_x = None
        self.var_y = None
        self.var_depth = None
        self.var_time = None
        self.time_units = None
        self.datetime_values = None
        self.var_mask = None
        self.num_x = None
        self.num_y = None
        self.num_times = None

    def get_valid_extent(self):
        """Masked model domain extent."""
        lon_min = numpy.nanmin(self.var_lon)
        lon_max = numpy.nanmax(self.var_lon)
        lat_min = numpy.nanmin(self.var_lat)
        lat_max = numpy.nanmax(self.var_lat)

        return lon_min, lon_max, lat_min, lat_max

    def init_handles(self):
        """Initialize handles to NetCDF variables."""
        self.var_lat = self.nc_file.variables['Latitude'][:, :]
        self.var_lon = self.nc_file.variables['Longitude'][:, :]
        self.var_lat = self.var_lat.astype(numpy.float64)
        self.var_lon = self.var_lon.astype(numpy.float64)
        self.var_u = self.nc_file.variables['u'][:, :, :, :]
        self.var_v = self.nc_file.variables['v'][:, :, :, :]
        self.var_date = self.nc_file.variables['Date'][:]
        self.var_x = self.nc_file.variables['X'][:]
        self.var_y = self.nc_file.variables['Y'][:]
        self.var_depth = self.nc_file.variables['Depth'][:]
        self.var_time = self.nc_file.variables['MT'][:]
        self.time_units = self.nc_file.variables['MT'].units
        self.num_times = self.nc_file.dimensions['MT'].size
        self.num_x = self.nc_file.dimensions['X'].size
        self.num_y = self.nc_file.dimensions['Y'].size
        # Use the surface layer in the u variable to define a land mask
        # Assuming this mask is sufficient for all variables, this may
        # not be the case for all hycom models
        self.var_mask = self.var_u[0, 0, :, :]

        self.update_datetime_values(netCDF4.num2date(self.var_time, units=self.time_units))

    def get_vertical_coordinate_type(self):
        pass

    def uv_to_regular_grid(self, model_index, time_index, target_depth, interp_method=interp.INTERP_METHOD_SCIPY):
        """Call grid processing functions and interpolate u/v to a regular grid"""

        u_target_depth, v_target_depth = vertical_interpolation(self.var_u, self.var_v, self.var_depth, self.num_x, self.num_y, time_index, target_depth)

        water_lat = numpy.ma.masked_array(self.var_lat, self.var_mask.mask)
        water_lon = numpy.ma.masked_array(self.var_lon, self.var_mask.mask)
        water_u = numpy.ma.masked_array(u_target_depth, self.var_mask.mask)
        water_v = numpy.ma.masked_array(v_target_depth, self.var_mask.mask)

        u_compressed = numpy.ma.compressed(water_u)
        v_compressed = numpy.ma.compressed(water_v)
        lat_compressed = numpy.ma.compressed(water_lat)
        lon_compressed = numpy.ma.compressed(water_lon)

        return interp.interpolate_to_regular_grid((u_compressed, v_compressed),
                                                  lon_compressed, lat_compressed,
                                                  model_index.var_x, model_index.var_y,
                                                  interp_method=interp_method)
        
    def output_native_grid(self, time_index, target_depth):
        """Generate output using native grid coordinates"""

        u_target_depth, v_target_depth = vertical_interpolation(self.var_u, self.var_v, self.var_depth, self.num_x, self.num_y, time_index, target_depth)

        water_lat = numpy.ma.masked_array(self.var_lat, self.var_mask.mask)
        water_lon = numpy.ma.masked_array(self.var_lon, self.var_mask.mask)
        water_u = numpy.ma.masked_array(u_target_depth, self.var_mask.mask)
        water_v = numpy.ma.masked_array(v_target_depth, self.var_mask.mask)

        u_compressed = numpy.ma.compressed(water_u)
        v_compressed = numpy.ma.compressed(water_v)
        lat_compressed = numpy.ma.compressed(water_lat)
        lon_compressed = numpy.ma.compressed(water_lon)

        return u_compressed, v_compressed, lat_compressed, lon_compressed


def vertical_interpolation(u, v, depth, num_x, num_y, time_index, target_depth):
    """Vertically interpolate variables to target depth.

    Args:
        u: `numpy.ndarray` containing u values for entire grid.
        v: `numpy.ndarray` containing v values for entire grid.
        depth: `numpy.1darray` containing depth in meters, positive down.
        time_index: Single forecast time index value.
        num_x: X dimensions
        num_y: Y dimensions
        target_depth: The target depth-below-sea-surface to which water
            currents will be interpolated, in meters. Must be zero or greater.
            For areas shallower than double this value, values will be
            interpolated to half the water column height instead. For
            navigationally significant currents, a value of 4.5 is recommended.
    """
    if target_depth < 0:
        raise Exception('Target depth must be positive')
    if target_depth > numpy.nanmax(depth):
        raise Exception('Target depth exceeds total depth')

    u_target_depth = numpy.ma.empty(shape=[num_y, num_x])
    v_target_depth = numpy.ma.empty(shape=[num_y, num_x])

    # For most models with standard depth levels ("z-levels"),
    # the vertical layer does not follow bathymetry, vertical
    # layers can be above and below the seafloor, therefore
    # the horizontal mask will vary by vertical layer

    # Determine the valid total depth at each
    # valid u/v horizontal location.
    for ny in range(num_y):
        for nx in range(num_x):
            deepest_valid_depth_layer_index = None
            for i in range(len(depth)):
                if u.mask[time_index, i, ny, nx] or v.mask[time_index, i, ny, nx]:
                    break
                deepest_valid_depth_layer_index = i
            if deepest_valid_depth_layer_index is None:
                continue

            # For areas shallower than the target depth, depth is half the total depth
            # Note that interp_depth is a positive value because the 'depth' variable
            # is positive-down.
            interp_depth = numpy.minimum(target_depth * 2, numpy.max(depth[deepest_valid_depth_layer_index])) / 2

            # Perform vertical linear interpolation on u/v values to target depth
            u_interp_depth = interpolate.interp1d(depth[:deepest_valid_depth_layer_index], u[time_index, :deepest_valid_depth_layer_index, ny, nx], fill_value='extrapolate')
            u_target_depth[ny, nx] = u_interp_depth(interp_depth)

            v_interp_depth = interpolate.interp1d(depth[:deepest_valid_depth_layer_index], v[time_index, :deepest_valid_depth_layer_index, ny, nx], fill_value='extrapolate')
            v_target_depth[ny, nx] = v_interp_depth(interp_depth)

    return u_target_depth, v_target_depth
