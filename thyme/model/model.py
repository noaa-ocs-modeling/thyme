"""
Utility classes and methods for working with hydrodynamic model output.

This module provides functionality allowing ocean model output variables to be
interpolated to a regular, orthogonal lat/lon horizontal grid, and supports
vertical interpolation of some variables (e.g. water currents) to a specified
depth-below-surface.

The classes in this module are abstract, meaning that they are intended to be
subclassed in order to support a distinct modeling system, as the details
involved with many of the processing steps are highly dependent on the
characteristics of the target hydrodynamic model.
"""
import json
import os

import netCDF4
import numpy
from osgeo import gdal, osr, ogr
from shapely.geometry import shape

from thyme.grid.regulargrid import RegularGrid
from thyme.util import dateutil

# Conversion factor for meters/sec to knots
MS2KNOTS = 1.943844

# Default fill value for NetCDF variables
FILLVALUE = -9999.0


class ModelIndexFile:
    """NetCDF file containing metadata/grid info used during model processing.

    Store a regular grid definition, mask, and other information needed to
    process/convert native output from a hydrodynamic model, within a reusable
    NetCDF file.

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

    Attributes:
        path: Path (relative or absolute) of the file..
        nc_file: Handle to `netCDF4.Dataset` instance for the opened NetCDF
            file.
        dim_x: Handle to x dimension.
        dim_y: Handle to y dimension.
        var_x: Handle to x coordinate variable (longitudes).
        var_y: Handle to y coordinate variable (latitudes).
        var_mask: Handle to master_mask variable.
    """
    DIMNAME_X = 'x'
    DIMNAME_Y = 'y'
    DIMNAME_SUBGRID = 'subgrid'

    def __init__(self, path):
        """Initialize ModelIndexFile object and open file at specified path.

        If file already exists and ``clobber==False``, it will be opened in
        read mode. Otherwise, it will be opened in write mode.
        
        Args:
            path: Path of target NetCDF file.
        """
        self.path = path
        self.nc_file = None
        self.dim_y = None
        self.dim_x = None
        self.dim_subgrid = None
        self.var_subgrid_id = None
        self.var_subgrid_name = None
        self.var_subgrid_x_min = None
        self.var_subgrid_x_max = None
        self.var_subgrid_y_min = None
        self.var_subgrid_y_max = None
        self.var_y = None
        self.var_x = None
        self.var_mask = None

    def open(self, clobber=False):
        """Open netCDF file.

        Args:
            clobber: (Optional, default False) If True, existing netCDF file at
                specified path, if any, will be deleted and the new file will
                be opened in write mode.
        """
        # nc_file: Handle to `netCDF4.Dataset` instance for the opened NetCDF
        if not os.path.exists(self.path) or clobber:
            self.nc_file = netCDF4.Dataset(self.path, 'w', format='NETCDF4')
        else:
            self.nc_file = netCDF4.Dataset(self.path, 'r', format='NETCDF4')
            self.init_handles()

    def close(self):
        """Close netCDF file."""
        self.nc_file.close()

    def init_handles(self):
        """Initialize handles to NetCDF dimensions/variables."""

        self.dim_y = self.nc_file.dimensions[self.DIMNAME_Y]
        self.dim_x = self.nc_file.dimensions[self.DIMNAME_X]

        try:
            self.dim_subgrid = self.nc_file.dimensions[self.DIMNAME_SUBGRID]
        except KeyError:
            self.dim_subgrid = None

        try:
            self.var_subgrid_id = self.nc_file.variables['subgrid_id'][:]
        except KeyError:
            self.var_subgrid_id = None
        try:
            self.var_subgrid_name = self.nc_file.variables['subgrid_name'][:]
        except KeyError:
            self.var_subgrid_name = None
        try:
            self.var_subgrid_x_min = self.nc_file.variables['subgrid_x_min'][:]
        except KeyError:
            self.var_subgrid_x_min = None
        try:
            self.var_subgrid_x_max = self.nc_file.variables['subgrid_x_max'][:]
        except KeyError:
            self.var_subgrid_x_max = None
        try:
            self.var_subgrid_y_min = self.nc_file.variables['subgrid_y_min'][:]
        except KeyError:
            self.var_subgrid_y_min = None
        try:
            self.var_subgrid_y_max = self.nc_file.variables['subgrid_y_max'][:]
        except KeyError:
            self.var_subgrid_y_max = None

        self.var_y = self.nc_file.variables[self.DIMNAME_Y][:]
        self.var_x = self.nc_file.variables[self.DIMNAME_X][:]
        self.var_mask = self.nc_file.variables['mask'][:, :]

    def create_dims_coord_vars(self, num_y, num_x):
        """Create index file NetCDF dimensions and coordinate variables.

        Args:
            num_y: Number of cells in x dimension.
            num_x: Number of cells in y dimension.
        """
        self.dim_y = self.nc_file.createDimension(self.DIMNAME_Y, num_y)
        self.dim_x = self.nc_file.createDimension(self.DIMNAME_X, num_x)

        # Create coordinate variables with same name as dimensions
        self.var_y = self.nc_file.createVariable(self.DIMNAME_Y, 'f4', (self.DIMNAME_Y,), fill_value=FILLVALUE)
        self.var_y.long_name = 'latitude of regular grid point at cell center'
        self.var_y.units = 'degree_north'
        self.var_y.standard_name = 'latitude'

        self.var_x = self.nc_file.createVariable(self.DIMNAME_X, 'f4', (self.DIMNAME_X,), fill_value=FILLVALUE)
        self.var_x.long_name = 'longitude of regular grid point at cell center'
        self.var_x.units = 'degree_east'
        self.var_x.standard_name = 'longitude'

        self.var_mask = self.nc_file.createVariable('mask', 'i4', (self.DIMNAME_Y, self.DIMNAME_X), fill_value=FILLVALUE)
        self.var_mask.long_name = 'regular grid point mask'
        self.var_mask.flag_values = -9999.0, 1
        self.var_mask.flag_meanings = 'land, water'

    def create_subgrid_dims_vars(self, num_subgrids, subset_grid_field_name=None):
        """Create subgrid-related NetCDF dimensions/variables.

        Args:
            num_subgrids: Number of subgrids.
            subset_grid_field_name: (Optional, default None) Shapefile field name
                to be stored in the index file.
        """
        self.dim_subgrid = self.nc_file.createDimension(self.DIMNAME_SUBGRID, num_subgrids)
        self.var_subgrid_id = self.nc_file.createVariable('subgrid_id', 'i4', (self.DIMNAME_SUBGRID,), fill_value=FILLVALUE)
        self.var_subgrid_x_min = self.nc_file.createVariable('subgrid_x_min', 'i4', (self.DIMNAME_SUBGRID,), fill_value=FILLVALUE)
        self.var_subgrid_x_max = self.nc_file.createVariable('subgrid_x_max', 'i4', (self.DIMNAME_SUBGRID,), fill_value=FILLVALUE)
        self.var_subgrid_y_min = self.nc_file.createVariable('subgrid_y_min', 'i4', (self.DIMNAME_SUBGRID,), fill_value=FILLVALUE)
        self.var_subgrid_y_max = self.nc_file.createVariable('subgrid_y_max', 'i4', (self.DIMNAME_SUBGRID,), fill_value=FILLVALUE)

        if subset_grid_field_name is not None:
            self.var_subgrid_name = self.nc_file.createVariable('subgrid_name', 'S30', (self.DIMNAME_SUBGRID,), fill_value=FILLVALUE)

    def init_nc(self, model_file, target_cellsize_meters, ofs_model, shoreline_shp=None, subset_grid_shp=None, subset_grid_field_name=None):
        """Initialize NetCDF dimensions/variables/attributes.

        Args:
            model_file: `ModelFile` instance containing model output used
                to identify properties of original grid.
            target_cellsize_meters: Target cell size of grid cells, in meters.
                Actual calculated x/y grid cell sizes will vary slightly from
                this value, since the regular grid uses lat/lon coordinates
                (thus a cell's width/height in meters will vary by latitude),
                and since it will be adjusted in order to fit a whole number of
                grid cells in the x and y directions within the calculated grid
                extent.
            ofs_model: The target model identifier.
            shoreline_shp: (Optional, default None) Path to a polygon shapefile
                containing features identifying land areas. If specified,
                a shoreline mask variable will be created/populated.
            subset_grid_shp: (Optional, default None) Path to a polygon
                shapefile containing orthogonal rectangles identifying areas
                to be used to subset (chop up) the full regular grid into
                tiles. Shapefile is assumed to be in the WGS84 projection. If
                None, the index file will be created assuming no subsets are
                desired and the extent of the model will be used instead.
            subset_grid_field_name: (Optional, default None) Shapefile
                field name to be stored in the index file.
        """
        # Calculate extent of valid (water) points
        (lon_min, lon_max, lat_min, lat_max) = model_file.get_valid_extent()

        # Index file global attributes
        self.nc_file.model = str.upper(ofs_model)
        self.nc_file.format = 'NetCDF-4'

        if shoreline_shp is not None:
            shoreline_path, shoreline = os.path.split(shoreline_shp)
            self.nc_file.shoreline = str(shoreline)

        if subset_grid_shp is not None:
            subgrid_path, subset_grid = os.path.split(subset_grid_shp)
            self.nc_file.subset_grid = str(subset_grid)

        # Populate grid x/y coordinate variables and subset-related variables
        # (if applicable)
        if subset_grid_shp is None:
            reg_grid = self.init_xy(lon_min, lat_min, lon_max, lat_max, target_cellsize_meters)
        elif subset_grid_field_name is None:
            reg_grid = self.init_xy_with_subsets(lon_min, lat_min, lon_max, lat_max, target_cellsize_meters,
                                                 subset_grid_shp)
        else:
            reg_grid = self.init_xy_with_subsets(lon_min, lat_min, lon_max, lat_max, target_cellsize_meters,
                                                 subset_grid_shp, subset_grid_field_name)
        # Index file global attributes
        self.nc_file.gridOriginLongitude = reg_grid.x_min
        self.nc_file.gridOriginLatitude = reg_grid.y_min

        land_mask = None
        if shoreline_shp is not None:
            land_mask = self.init_shoreline_mask(reg_grid, shoreline_shp)

        print('Grid dimensions (y,x): ({},{})'.format(len(reg_grid.y_coords), len(reg_grid.x_coords)))

        # Calculate the mask
        grid_cell_mask = self.compute_grid_mask(model_file, reg_grid)
        self.write_mask(land_mask, grid_cell_mask)

    def init_xy(self, lon_min, lat_min, lon_max, lat_max, target_cellsize_meters):
        """Create & initialize x/y dimensions/coordinate vars.
        
        Args:
            lon_min: Minimum longitude of domain.
            lat_min: Minimum latitude of domain.
            lon_max: Maximum longitude of domain.
            lat_max: Maximum latitude of domain.
            target_cellsize_meters: Target cell size, in meters. Actual
                calculated cell sizes will be approximations of this.
        """
        # Calculate actual x/y cell sizes
        cellsize_x, cellsize_y = RegularGrid.calc_cellsizes(lon_min, lat_min, lon_max, lat_max, target_cellsize_meters)
        
        # Build a regular grid using calculated cell sizes and given extent
        reg_grid = RegularGrid(lon_min, lat_min, lon_max, lat_max, cellsize_x, cellsize_y)
        
        # Create NetCDF dimensions & coordinate variables using dimension sizes
        # from regular grid
        self.create_dims_coord_vars(len(reg_grid.y_coords), len(reg_grid.x_coords))
        
        # Populate NetCDF coordinate variables using regular grid coordinates
        self.var_x[:] = reg_grid.x_coords[:]
        self.var_y[:] = reg_grid.y_coords[:]
        self.nc_file.gridSpacingLongitude = cellsize_x
        self.nc_file.gridSpacingLatitude = cellsize_y

        return reg_grid

    def init_xy_with_subsets(self, lon_min, lat_min, lon_max, lat_max, target_cellsize_meters, subset_grid_shp, subset_grid_field_name=None):
        """Create & initialize x/y dimensions/coordinate vars and subset vars.

        Args:
            lon_min: Minimum longitude of domain.
            lat_min: Minimum latitude of domain.
            lon_max: Maximum longitude of domain.
            lat_max: Maximum latitude of domain.
            target_cellsize_meters: Target cell size, in meters. Actual
                calculated cell sizes will be approximations of this.
            subset_grid_shp: Path to subset grid polygon shapefile used to
                define subgrid domains.
            subset_grid_field_name: Optional, default None) Shapefile
                field name to be stored in the index file.

        Raises: Exception when given subset grid shapefile does not exist or
            does not include any grid polygons intersecting with given extent.

        Returns:
            Instance of `RegularGrid` representing the extended generated
            grid whose extent matches the union of all intersecting subset grid
            polygons.
        """
        shp = ogr.Open(subset_grid_shp)
        layer = shp.GetLayer()

        # Create OGR Geometry from ocean model grid extent
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(lon_min, lat_max)
        ring.AddPoint(lon_min, lat_min)
        ring.AddPoint(lon_max, lat_min)
        ring.AddPoint(lon_max, lat_max)
        ring.AddPoint(lon_min, lat_max)
        # Create polygon
        ofs_poly = ogr.Geometry(ogr.wkbPolygon)
        ofs_poly.AddGeometry(ring)

        # Get the EPSG value from the import shapefile and transform to WGS84
        spatial_ref = layer.GetSpatialRef()
        shp_srs = spatial_ref.GetAttrValue('AUTHORITY', 1)
        source = osr.SpatialReference()
        source.ImportFromEPSG(int(shp_srs))
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(source, target)
        ofs_poly.Transform(transform)

        # Find the intersection between grid polygon and ocean model grid extent
        subset_polys = {}
        fids = []
        fields = {}
        fid = 0
        for feature in layer:
            geom = feature.GetGeometryRef()
            if ofs_poly.Intersects(geom):
                subset_polys[fid] = geom.ExportToJson()
                if subset_grid_field_name is not None:
                    field_name = feature.GetField(str(subset_grid_field_name))
                    fields.update({fid: field_name})
                    fids.append(fid)
                else:
                    fids.append(fid)
            fid += 1

        if len(fids) == 0:
            raise Exception('Given subset grid shapefile contains no polygons that intersect with model domain; cannot proceed.')

        # Use a single subset polygon to calculate x/y cell sizes. This ensures
        # that cells do not fall on the border between two grid polygons.
        single_polygon = ogr.Geometry(ogr.wkbMultiPolygon)
        single_polygon.AddGeometry(ogr.CreateGeometryFromJson(subset_polys[fids[0]]))
        sp_x_min, sp_x_max, sp_y_min, sp_y_max = single_polygon.GetEnvelope()

        cellsize_x, cellsize_y = RegularGrid.calc_cellsizes(sp_x_min, sp_y_min, sp_x_max, sp_y_max, target_cellsize_meters)

        # Combine identified subset grid polygons into single multipolygon to
        # calculate full extent of all combined subset grids
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for fid in fids:
            multipolygon.AddGeometry(ogr.CreateGeometryFromJson(subset_polys[fid]))

        (x_min, x_max, y_min, y_max) = multipolygon.GetEnvelope()
        full_reg_grid = RegularGrid(x_min, y_min, x_max, y_max, cellsize_x, cellsize_y)

        # Create NetCDF dimensions & coordinate variables using dimension sizes
        # from regular grid
        self.create_dims_coord_vars(len(full_reg_grid.y_coords), len(full_reg_grid.x_coords))
        # Populate NetCDF coordinate variables using regular grid coordinates
        self.var_x[:] = full_reg_grid.x_coords[:]
        self.var_y[:] = full_reg_grid.y_coords[:]
        self.nc_file.gridSpacingLongitude = full_reg_grid.cellsize_x
        self.nc_file.gridSpacingLatitude = full_reg_grid.cellsize_y

        # Create subgrid dimension/variables
        self.create_subgrid_dims_vars(len(subset_polys), subset_grid_field_name)
        # Calculate subgrid mask ranges, populate subgrid ID
        for subgrid_index, fid in enumerate(fids):
            self.var_subgrid_id[subgrid_index] = fid
            if subset_grid_field_name is not None:
                self.var_subgrid_name[subgrid_index] = fields[fid]

            # Convert OGR geometry to shapely geometry
            subset_poly_shape = shape(json.loads(subset_polys[fid]))
            min_x_coord = subset_poly_shape.bounds[0]
            max_x_coord = subset_poly_shape.bounds[2]
            min_y_coord = subset_poly_shape.bounds[1]
            max_y_coord = subset_poly_shape.bounds[3]

            subgrid_x_min = None
            subgrid_x_max = None
            subgrid_y_min = None
            subgrid_y_max = None

            for i, x in enumerate(self.var_x):
                if x >= min_x_coord:
                    subgrid_x_min = i
                    break
            count_x = round(((max_x_coord - min_x_coord) / full_reg_grid.cellsize_x))

            for i, y in enumerate(self.var_y):
                if y >= min_y_coord:
                    subgrid_y_min = i
                    break
            count_y = round(((max_y_coord - min_y_coord) / full_reg_grid.cellsize_y))

            subgrid_x_max = subgrid_x_min + count_x - 1
            subgrid_y_max = subgrid_y_min + count_y - 1

            self.var_subgrid_x_min[subgrid_index] = subgrid_x_min
            self.var_subgrid_x_max[subgrid_index] = subgrid_x_max
            self.var_subgrid_y_min[subgrid_index] = subgrid_y_min
            self.var_subgrid_y_max[subgrid_index] = subgrid_y_max

        return full_reg_grid

    @staticmethod
    def init_shoreline_mask(reg_grid, shoreline_shp):
        """Create a shoreline mask for region of interest at target resolution.
        
        Args:
            reg_grid: `RegularGrid` instance describing the regular grid for
                which the shoreline mask will be created.
            shoreline_shp: Path to a polygon shapefile containing features
                identifying land areas.

        Returns:
            2D numpy array, matching the dimensions of the given RegularGrid,
            containing a value of 0 for water areas and a value of 2 for land
            areas.
        """
        driver = ogr.GetDriverByName("ESRI Shapefile")
        shp = driver.Open(shoreline_shp)
        layer = shp.GetLayer()

        # Rasterize the shoreline polygon layer and write to memory
        # Optionally write to gdal.GetDriverByName('GTIFF') for output a GeoTIFF
        pixel_width = reg_grid.cellsize_x
        pixel_height = reg_grid.cellsize_y
        cols = len(reg_grid.y_coords)
        rows = len(reg_grid.x_coords)
        target_dset = gdal.GetDriverByName('MEM').Create('land_mask.tif', rows, cols, 1, gdal.GDT_Byte)
        target_dset.SetGeoTransform((reg_grid.x_min, pixel_width, 0, reg_grid.y_min, 0,  pixel_height))
        target_dset_srs = osr.SpatialReference()
        target_dset_srs.ImportFromEPSG(4326)
        target_dset.SetProjection(target_dset_srs.ExportToWkt())
        band = target_dset.GetRasterBand(1)
        band.SetNoDataValue(FILLVALUE)
        band.FlushCache()

        gdal.RasterizeLayer(target_dset, [1], layer, burn_values=[2])

        # Store as a numpy array, land = 2, valid areas = 0
        target_band = target_dset.GetRasterBand(1)
        land_mask = target_band.ReadAsArray(pixel_width, pixel_height, rows, cols).astype(numpy.int)
        del target_dset

        return land_mask

    def compute_grid_mask(self, model_file, reg_grid):
        """Create regular grid model domain mask.

        The model domain mask identifies which output grid cells may contain
        valid data based on the model's grid coverage (however, this mask
        should be combined with a shoreline mask to create a combined mask).

        This function should be overridden by the subclass, as its
        implementation is very specific to the characteristics of the native
        model grid.

        Args:
            model_file: `ModelFile` instance containing model native grid
                structure variables.
            reg_grid: `RegularGrid` instance describing the regular grid for
                which the mask will be created.

        Raises:
            `NotImplementedError`: If subclass has not overridden this method.
        """
        raise NotImplementedError("model.compute_grid_mask() must be overridden by subclass")

    @staticmethod
    def rasterize_mask(reg_grid, layer):
        """Rasterize model domain mask from native grid cell polygons.

        This creates an in-memory representation 

        Args:
            reg_grid: `RegularGrid` instance describing the regular grid for
                which the mask will be created.
            layer: `osgeo.ogr.Layer` instance containing polygon features
                representing valid data areas. Should be derived from the
                native model grid based on its characteristics (e.g. irregular
                vs unstructured grid cells, internal model grid mask, etc.).

        Returns:
            `numpy` `int` array, corresponding with specified
            `RegularGrid`, identifying valid and not-valid output grid cells.
            Valid cells are set to a value of ``1`` while invalid cells are set
            to ``0``.
        """
        # Rasterize the grid cell polygon layer and write to memory
        # Optionally write to gdal.GetDriverByName('GTIFF') for output a GeoTIFF
        pixel_width = reg_grid.cellsize_x
        pixel_height = reg_grid.cellsize_y
        cols = len(reg_grid.y_coords)
        rows = len(reg_grid.x_coords)
        target_dset = gdal.GetDriverByName('MEM').Create('grid_cell_mask.tif', rows, cols, 1, gdal.GDT_Byte)
        target_dset.SetGeoTransform((reg_grid.x_min, pixel_width, 0, reg_grid.y_min, 0, pixel_height))
        target_dset_srs = osr.SpatialReference()
        target_dset_srs.ImportFromEPSG(4326)
        target_dset.SetProjection(target_dset_srs.ExportToWkt())
        band = target_dset.GetRasterBand(1)
        band.SetNoDataValue(FILLVALUE)
        band.FlushCache()

        gdal.RasterizeLayer(target_dset, [1], layer, burn_values=[1])

        # Store as numpy array, valid areas = 1, invalid areas = 0
        target_band = target_dset.GetRasterBand(1)
        grid_cell_mask = target_band.ReadAsArray(pixel_width, pixel_height, rows, cols).astype(numpy.int)

        return grid_cell_mask

    def write_mask(self, land_mask, grid_cell_mask):
        """Write master mask to index file.

        Args:
            grid_cell_mask: 2D numpy array, matching the index file dimensions,
                containing regular grid model domain mask values, a value of 1
                for valid water, a value of 0 for invalid areas.
            land_mask: 2D numpy array, matching the index file dimensions,
                containing regular grid masked values, a value of 0 for water
                and a value of 2 for land.
        """
        # Use land mask and grid cell mask to create a master mask
        # Value of 1 for valid areas, FILLVALUE for invalid areas

        # Write mask to index file
        for y in range(self.dim_y.size):
            for x in range(self.dim_x.size):
                if land_mask is not None and land_mask[y, x] != 0:
                    continue
                if grid_cell_mask[y, x] == 1:
                    self.var_mask[y, x] = 1
                else:
                    self.var_mask[y, x] = FILLVALUE


class ModelFile:
    """Read/process data from a numerical ocean model file.

    This is an abstract base class that should be inherited from.

    Opens a NetCDF model file, reads variables, gets model domain extent,
    converts values, and interpolates model variables to a regular grid.
    """
    def __init__(self, path, datetime_rounding=None):
        """Initialize model file object and opens file at specified path.

        Args:
            path: Path of target NetCDF file.
            datetime_rounding: The `dateutil.DatetimeRounding` constant
                representing how date/time values should be rounded, or None if
                no rounding should occur.
        """
        self.path = path
        self.datetime_rounding = datetime_rounding
        self.datetime_values = None
        self.nc_file = None

    def open(self):
        """Open the model output file in read mode.

        Raises:
            Exception: If specified NetCDF file does not exist.
        """
        if os.path.exists(self.path):
            self.nc_file = netCDF4.Dataset(self.path, 'r', format='NETCDF3_CLASSIC')
            self.init_handles()
        else:
            # File doesn't exist, raise error
            raise(Exception('NetCDF file does not exist: {}'.format(self.path)))

    def close(self):
        """Close the model output file & release resources."""
        self.nc_file.close()
        self.release_resources()

    def release_resources(self):
        """Allow GC to reclaim memory by releasing/deleting resources."""
        raise NotImplementedError("model.release_resources() must be overridden by subclass")

    def get_valid_extent(self):
        raise NotImplementedError("model.get_valid_extent() must be overridden by subclass")

    def init_handles(self):
        raise NotImplementedError("model.init_handles() must be overridden by subclass")

    def get_vertical_coordinate_type(self):
        raise NotImplementedError("model.get_vertical_coordinate_type() must be overridden by subclass")

    def update_datetime_values(self, datetimes):
        """Update datetime values, rounding them if configured to do so.

        Args:
            datetimes: List of new, unrounded `datetime` values.
        """
        self.datetime_values = []
        for i in range(len(datetimes)):
            self.datetime_values.append(dateutil.round(datetimes[i], self.datetime_rounding))

    def uv_to_regular_grid(self, model_index, time_index, target_depth, interp_method=None):
        """Interpolate u/v current velocity components to regular grid.
        
        This function should be overridden, as its implementation is very
        specific to the characteristics of the native model output.
        """
        raise NotImplementedError("model.uv_to_regular_grid() must be overridden by subclass")

    def output_native_grid(self, time_index, target_depth):
        """Generate output using native grid coordinates

        This function should be overridden, as its implementation is very
        specific to the characteristics of the native model output.
        """
        raise NotImplementedError("model.output_native_grid() must be overridden by subclass")


def irregular_uv_to_speed_direction(u, v):
    """Convert u and v vectors to speed/direction.

    Input u/v values are assumed to be in meters/sec. Output speed values will
    be converted to knots and direction in degrees from true north (0-360).

    Args:
        u: `numpy.ma.masked_array` containing 1-dimension u values
        v: `numpy.ma.masked_array` containing 1-dimension v values

    Returns:
        One-tuple containing the 2D `numpy.ma.masked_array`s for direction
        and speed (in that order).
    """
    direction = numpy.ma.empty(v.shape, dtype=numpy.float32)
    speed = numpy.ma.empty(u.shape, dtype=numpy.float32)

    for i in numpy.ndindex(speed.shape):
        u_ms = u[i]
        v_ms = v[i]

        u_knot = u_ms * MS2KNOTS
        v_knot = v_ms * MS2KNOTS

        current_speed = numpy.sqrt(u_knot**2 + v_knot**2)
        current_direction_radians = numpy.arctan2(v_knot, u_knot)
        current_direction_degrees = numpy.degrees(current_direction_radians)
        current_direction_north = 90.0 - current_direction_degrees

        if current_direction_north < 0.0:
            current_direction_north += 360.0

        direction[i] = current_direction_north
        speed[i] = current_speed

    return speed, direction


def regular_uv_to_speed_direction(reg_grid_u, reg_grid_v):
    """Convert u and v vectors to speed/direction.

    Input u/v values are assumed to be in meters/sec. Output speed values will
    be converted to knots and direction in degrees from true north (0-360).

    Args:
        reg_grid_u: `numpy.ma.masked_array` containing u values interpolated
            to the regular grid.
        reg_grid_v: `numpy.ma.masked_array` containing v values interpolated
            to the regular grid.

    Returns:
        Two-tuple containing the 2D `numpy.ma.masked_array`s for direction
        and speed (in that order).
    """
    direction = numpy.ma.empty(reg_grid_v.shape, dtype=numpy.float32)
    speed = numpy.ma.empty(reg_grid_u.shape, dtype=numpy.float32)

    for y, x in numpy.ndindex(speed.shape):
        if reg_grid_u.mask[y, x]:
            direction[y, x] = numpy.nan
            speed[y, x] = numpy.nan
            continue

        u_ms = reg_grid_u[y, x]
        v_ms = reg_grid_v[y, x]

        u_knot = u_ms * MS2KNOTS
        v_knot = v_ms * MS2KNOTS

        current_speed = numpy.sqrt(u_knot**2 + v_knot**2)
        current_direction_radians = numpy.arctan2(v_knot, u_knot)
        current_direction_degrees = numpy.degrees(current_direction_radians)
        current_direction_north = 90.0 - current_direction_degrees

        if current_direction_north < 0.0:
            current_direction_north += 360.0

        direction[y, x] = current_direction_north
        speed[y, x] = current_speed

    return speed, direction



