"""
Utility classes and methods for working with a rectilinear grid output.

This module provides functionality allowing model output to be interpolated to
a regular, orthogonal lat/lon horizontal grid at a given depth-below-surface or
to be optimized for encoding in a different format.

"""
import netCDF4
import numpy
from osgeo import ogr, osr

from thyme.model import model
from thyme.util import interp

# Default fill value for NetCDF variables
FILLVALUE = -9999.0


class RectilinearIndexFile(model.ModelIndexFile):
    """NetCDF file containing metadata/grid info used during processing.

    Store a regular grid definition, mask, and other information needed to
    process/convert native output from a hydrodynamic model,
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
            model_file: `RectilinearOutputFile` instance containing rectlinear
                grid structure and variables.
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

        # Create polygons for each rectilinear grid cell
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


class RectilinearFile(model.ModelFile):
    """Read/process data from a rectilinear model NetCDF file.

    Attributes:
        path: Path (relative or absolute) of the file.
    """

    def __init__(self, path, datetime_rounding=None):
        """Initialize rectilinear file object and open file at specified path.

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
        self.var_lat = self.nc_file.variables['lat'][:, :]
        self.var_lon = self.nc_file.variables['lon'][:, :]
        self.var_lat = self.var_lat.astype(numpy.float64)
        self.var_lon = self.var_lon.astype(numpy.float64)
        self.var_u = self.nc_file.variables['u'][:, :, :, :]
        self.var_v = self.nc_file.variables['v'][:, :, :, :]
        self.var_depth = self.nc_file.variables['depth'][:]
        self.var_time = self.nc_file.variables['time'][:]
        self.time_units = self.nc_file.variables['time'].units
        self.num_times = self.nc_file.dimensions['time'].size
        self.num_x = self.nc_file.dimensions['X'].size
        self.num_y = self.nc_file.dimensions['Y'].size

        # Use the surface layer in the u variable to define a land mask
        mask_u = self.var_u[0, 0, :, :]
        mask_v = self.var_v[0, 0, :, :]
        self.var_mask = numpy.ma.mask_or(mask_u.mask, mask_v.mask)

        self.update_datetime_values(netCDF4.num2date(self.var_time, units=self.time_units))

    def get_vertical_coordinate_type(self):
        pass

    def uv_to_regular_grid(self, model_index, time_index, target_depth, interp_method=interp.INTERP_METHOD_SCIPY):
        """Call grid processing functions and interpolate u/v to a regular grid"""

        u_single = self.var_u[time_index, 0, :, :]
        v_single = self.var_v[time_index, 0, :, :]

        water_lat = numpy.ma.masked_array(self.var_lat, self.var_mask,)
        water_lon = numpy.ma.masked_array(self.var_lon, self.var_mask,)
        water_u = numpy.ma.masked_array(u_single, self.var_mask,)
        water_v = numpy.ma.masked_array(v_single, self.var_mask,)

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

        u_single = self.var_u[time_index, 0, :, :]
        v_single = self.var_v[time_index, 0, :, :]

        water_lat = numpy.ma.masked_array(self.var_lat, self.var_mask,)
        water_lon = numpy.ma.masked_array(self.var_lon, self.var_mask,)
        water_u = numpy.ma.masked_array(u_single, self.var_mask,)
        water_v = numpy.ma.masked_array(v_single, self.var_mask,)

        u_compressed = numpy.ma.compressed(water_u)
        v_compressed = numpy.ma.compressed(water_v)
        lat_compressed = numpy.ma.compressed(water_lat)
        lon_compressed = numpy.ma.compressed(water_lon)

        return u_compressed, v_compressed, lat_compressed, lon_compressed


