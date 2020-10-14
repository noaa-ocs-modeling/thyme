"""
Utility classes and methods for working with ADCIRC output to derive water levels.

ADvanced CIRculation Model (ADCIRC) is a two-dimensional, depth-integrated with
internal tides and spherical coordinates on an unstructured global mesh.

This module provides functionality allowing ADCIRC output to be interpolated to
a regular, orthogonal lat/lon horizontal grid.

Currently, this module has only been tested to work with the ADCIRC-based Global
Extra-Tropical Surge and Tide Operational Forecast System (G-ESTOFS) produced by NOS,
and would likely require modifications to support other ADCIRC-based model output.
"""
import netCDF4
import numpy
from osgeo import ogr, osr

from thyme.model import model
from thyme.util import interp

FILLVALUE = 0


class ADCIRCIndexFile(model.ModelIndexFile):
    """NetCDF file containing metadata/grid info used during ADCIRC processing.

    Store a regular grid definition, mask, and other information needed to
    process/convert native output from an ADCIRC-based hydrodynamic model,
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

        For every centroid create a polygon from three valid elements.
        Rasterize the polygon to create a grid domain mask.

        Args:
            model_file: `ADCIRCFile` instance containing unstructured grid
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

        # Create polygons for each triangular unstructured network
        # using three valid elements surrounding each node
        for nele in range(0, model_file.var_element.shape[0]):
            p1 = model_file.var_element[nele][0] - 1
            p2 = model_file.var_element[nele][1] - 1
            p3 = model_file.var_element[nele][2] - 1
            element_lon = [model_file.var_lon[p1], model_file.var_lon[p2], model_file.var_lon[p3]]
            if all(i >= 0 for i in element_lon) or all(i < 0 for i in element_lon) \
                    or any(-25 <= i <= 25 for i in element_lon):
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(model_file.var_lon[p1], model_file.var_lat[p1])
                ring.AddPoint(model_file.var_lon[p2], model_file.var_lat[p2])
                ring.AddPoint(model_file.var_lon[p3], model_file.var_lat[p3])
                ring.AddPoint(model_file.var_lon[p1], model_file.var_lat[p1])
                # Create polygon
                geom = ogr.Geometry(ogr.wkbPolygon)
                geom.AddGeometry(ring)
                feat = ogr.Feature(layer.GetLayerDefn())
                feat.SetField('id', 1)
                feat.SetGeometry(geom)
                layer.CreateFeature(feat)
                feat = None

        return self.rasterize_mask(reg_grid, layer)


class ADCIRCFile(model.ModelFile):
    """Read/process data from a ADCIRC model NetCDF file.

    Attributes:
        path: Path (relative or absolute) of the file.
    """

    def __init__(self, path, file_object=None, datetime_rounding=None):
        """Initialize and open ADCIRC file object at specified path.

        Args:
            path: Path of target NetCDF file.
            file_object: Memory or disk based NetCDF file object
            datetime_rounding: The `dateutil.DatetimeRounding` constant
                representing how date/time values should be rounded, or None if
                no rounding should occur.
        """
        super().__init__(path, file_object=file_object, datetime_rounding=datetime_rounding)
        self.var_lon = None
        self.var_lat = None
        self.var_zeta = None
        self.var_depth = None
        self.var_element = None
        self.var_time = None
        self.time_units = None
        self.num_nodes = None
        self.num_times = None
        self.num_x = None
        self.num_y = None

    def release_resources(self):
        """Allow GC to reclaim memory by releasing/deleting resources."""
        self.var_lon = None
        self.var_lat = None
        self.var_zeta = None
        self.var_depth = None
        self.var_element = None
        self.var_time = None
        self.time_units = None
        self.num_nodes = None
        self.num_times = None
        self.num_x = None
        self.num_y = None

    def get_valid_extent(self):
        """Masked model domain extent."""
        lon_min = numpy.nanmin(self.var_lon)
        lon_max = numpy.nanmax(self.var_lon)
        lat_min = numpy.nanmin(self.var_lat)
        lat_max = numpy.nanmax(self.var_lat)

        return lon_min, lon_max, lat_min, lat_max

    def init_handles(self):
        """Initialize handles to NetCDF variables."""
        self.var_lon = self.nc_file.variables['x'][:]
        self.var_lat = self.nc_file.variables['y'][:]
        self.var_zeta = self.nc_file.variables['zeta'][:, :]
        self.var_depth = self.nc_file.variables['depth'][:]
        self.var_element = self.nc_file.variables['element'][:, :]
        self.num_nodes = self.nc_file.dimensions['node'].size
        self.var_time = self.nc_file.variables['time'][:]
        self.var_time = self.var_time[7:]  # skip 6 hour nowcast
        self.time_units = self.nc_file.variables['time'].units
        self.num_times = self.nc_file.dimensions['time'].size
        self.num_x = self.nc_file.variables['x'].size
        self.num_y = self.nc_file.variables['y'].size

        self.update_datetime_values(netCDF4.num2date(self.var_time, units=self.time_units))

    def get_vertical_coordinate_type(self):
        pass

    def output_regular_grid(self, model_index, time_index, target_depth=None, interp_method=interp.INTERP_METHOD_SCIPY):
        """Call grid processing functions and interpolate target variable to a regular grid"""

        return interp.interpolate_to_regular_grid((self.var_zeta[time_index, :],), self.var_lon, self.var_lat, model_index.var_x,
                                                  model_index.var_y, interp_method=interp_method)

    def output_native_grid(self, time_index, target_depth=None):
        """Generate output using native grid coordinates"""

        return self.var_zeta[time_index], self.var_lat, self.var_lon

