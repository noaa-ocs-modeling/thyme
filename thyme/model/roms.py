"""
Utility classes and methods for working with ROMS output.

The Regional Ocean Modeling System (ROMS) is "a free-surface, terrain-
following, primitive equations ocean model widely used by the scientific
community for a diverse range of applications." ROMS uses an irregular,
curvilinear horizontal grid and a sigma (terrain-following) vertical coordinate
system. See https://www.myroms.org for more information.

This module provides functionality allowing ROMS output to be interpolated to a
regular, orthogonal lat/lon horizontal grid at a given depth-below-surface.

Currently, this module has only been tested to work with ROMS-based National
Ocean Service (NOS) Operational Forecast Systems (OFS), e.g. CBOFS, DBOFS,
TBOFS, and GoMOFS, and would likely require modifications to support other
ROMS-based model output.
"""
import netCDF4
import numpy
from osgeo import ogr, osr
from scipy import interpolate

from thyme.model import model
from thyme.util import interp

# Default fill value for NetCDF variables
FILLVALUE = -9999.0


class ROMSIndexFile(model.ModelIndexFile):
    """NetCDF file containing metadata/grid info used during ROMS processing.

    Store a regular grid definition, mask, and other information needed to
    process/convert native output from an ROMS-based hydrodynamic model, within
    a reusable NetCDF file.

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

        For every irregular grid point create a polygon from four valid
        grid points, searching counter clockwise (eta1,xi1), (eta2,xi2),
        (eta3,xi3), (eta4,xi4). Rasterize the polygon to create a grid domain
        mask.

        Args:
            model_file: ``ROMSOutputFile`` instance containing irregular grid
                structure and variables.
            reg_grid: ``RegularGrid`` instance describing the regular grid for
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
        for xi1 in range(model_file.num_xi - 1):
            for eta1 in range(model_file.num_eta - 1):
                xi2 = xi1 + 1
                eta2 = eta1
                xi3 = xi1 + 1
                eta3 = eta1 + 1
                xi4 = xi1
                eta4 = eta1 + 1

                # Search valid points
                valid_points = []
                for (eta, xi) in ((eta1, xi1), (eta2, xi2), (eta3, xi3), (eta4, xi4)):
                    if model_file.var_mask_rho[eta, xi] == 1:
                        valid_points.append((eta, xi))
                if len(valid_points) < 3:
                    continue
                ring = ogr.Geometry(ogr.wkbLinearRing)
                for (eta, xi) in valid_points:
                    ring.AddPoint(model_file.var_lon_rho[eta, xi], model_file.var_lat_rho[eta, xi])
                (eta, xi) = valid_points[0]
                ring.AddPoint(model_file.var_lon_rho[eta, xi], model_file.var_lat_rho[eta, xi])

                # Create polygon
                geom = ogr.Geometry(ogr.wkbPolygon)
                geom.AddGeometry(ring)
                feat = ogr.Feature(layer.GetLayerDefn())
                feat.SetField('id', 1)
                feat.SetGeometry(geom)
                layer.CreateFeature(feat)

        return self.rasterize_mask(reg_grid, layer)


class ROMSFile(model.ModelFile):
    """Read/process data from a ROMS model NetCDF file.

    Attributes:
        path: Path (relative or absolute) of the file.
    """
    def __init__(self, path, datetime_rounding=None):
        """Initialize ROMS file object and open file at specified path.

        Args:
            path: Path of target NetCDF file.
            datetime_rounding: The `dateutil.DatetimeRounding` constant
                representing how date/time values should be rounded, or None if
                no rounding should occur.
        """
        super().__init__(path, datetime_rounding=datetime_rounding)
        self.var_ang_rho = None
        self.var_lat_rho = None
        self.var_lon_rho = None
        self.var_u = None
        self.var_v = None
        self.var_mask_u = None
        self.var_mask_v = None
        self.var_mask_rho = None
        self.var_zeta = None
        self.var_h = None
        self.var_s_rho = None
        self.var_hc = None
        self.var_cs_r = None
        self.var_vtransform = None
        self.var_time = None
        self.time_units = None
        self.num_eta = None
        self.num_xi = None
        self.num_sigma = None
        self.num_times = None
        self.datetime_values = None

    def release_resources(self):
        """Allow GC to reclaim memory by releasing/deleting resources."""
        self.var_ang_rho = None
        self.var_lat_rho = None
        self.var_lon_rho = None
        self.var_u = None
        self.var_v = None
        self.var_mask_u = None
        self.var_mask_v = None
        self.var_mask_rho = None
        self.var_zeta = None
        self.var_h = None
        self.var_s_rho = None
        self.var_hc = None
        self.var_cs_r = None
        self.var_vtransform = None
        self.var_time = None
        self.time_units = None
        self.num_eta = None
        self.num_xi = None
        self.num_sigma = None
        self.num_times = None
        self.datetime_values = None

    def get_valid_extent(self):
        """Masked model domain extent."""
        water_lat_rho = numpy.ma.masked_array(self.var_lat_rho, numpy.logical_not(self.var_mask_rho))
        water_lon_rho = numpy.ma.masked_array(self.var_lon_rho, numpy.logical_not(self.var_mask_rho))
        lon_min = numpy.nanmin(water_lon_rho)
        lon_max = numpy.nanmax(water_lon_rho)
        lat_min = numpy.nanmin(water_lat_rho)
        lat_max = numpy.nanmax(water_lat_rho)

        return lon_min, lon_max, lat_min, lat_max

    def init_handles(self):
        """Initialize handles to NetCDF variables.

        Because there is no U for the first column (xi=0) and no V for the
        first row (eta=0) in the ROMS staggered grid, we must skip the first
        row and column of every variable associated with rho points as well as
        the first row (eta=0) of U values, and the first column (xi=0) of V
        values. This ensures that u, v, and all other rho variables have the
        same dimensions in xi and eta, with u[eta,xi], v[eta,xi], and rho
        variables e.g. ang_rho[eta,xi] all correspond with the same grid cell
        for a given [eta,xi] coordinate.
        """
        self.var_ang_rho = self.nc_file.variables['angle'][1:, 1:]
        self.var_lat_rho = self.nc_file.variables['lat_rho'][1:, 1:]
        self.var_lon_rho = self.nc_file.variables['lon_rho'][1:, 1:]
        self.var_u = self.nc_file.variables['u'][:, :, 1:, :]
        self.var_v = self.nc_file.variables['v'][:, :, :, 1:]
        self.var_mask_u = self.nc_file.variables['mask_u'][1:, :]
        self.var_mask_v = self.nc_file.variables['mask_v'][:, 1:]
        self.var_mask_rho = self.nc_file.variables['mask_rho'][1:, 1:]
        self.var_zeta = self.nc_file.variables['zeta'][:, 1:, 1:]
        self.var_h = self.nc_file.variables['h'][1:, 1:]
        self.var_s_rho = self.nc_file.variables['s_rho'][:]
        self.var_hc = self.nc_file.variables['hc'][:]
        self.var_cs_r = self.nc_file.variables['Cs_r'][:]
        self.var_vtransform = self.nc_file.variables['Vtransform'][:]
        self.var_time = self.nc_file.variables['ocean_time'][:]
        self.time_units = self.nc_file.variables['ocean_time'].units
        self.num_eta = self.nc_file.dimensions['eta_rho'].size - 1
        self.num_xi = self.nc_file.dimensions['xi_rho'].size - 1
        self.num_sigma = self.nc_file.dimensions['s_rho'].size
        self.num_times = self.nc_file.dimensions['ocean_time'].size

        self.update_datetime_values(netCDF4.num2date(self.var_time, units=self.time_units, calendar='proleptic_gregorian'))

        # Determine if sigma values are positive up or down from netCDF metadata
        if self.nc_file.variables['s_rho'].positive == 'down':
            self.var_s_rho = self.var_s_rho * -1

    def get_vertical_coordinate_type(self):
        pass

    def uv_to_regular_grid(self, model_index, time_index, target_depth, interp_method=interp.INTERP_METHOD_SCIPY):
        """Call grid processing functions and interpolate averaged, rotated u/v to a regular grid"""

        u_target_depth, v_target_depth = vertical_interpolation(self.var_u, self.var_v, self.var_s_rho,
                                                                self.var_mask_rho, self.var_mask_u, self.var_mask_v,
                                                                self.var_zeta, self.var_h, self.var_hc, self.var_cs_r,
                                                                self.var_vtransform, self.num_eta, self.num_xi,
                                                                self.num_sigma, time_index, target_depth)

        # Create masked arrays to mask land values
        water_u = numpy.ma.masked_array(u_target_depth, numpy.logical_not(self.var_mask_u))
        water_v = numpy.ma.masked_array(v_target_depth, numpy.logical_not(self.var_mask_v))

        # u/v masked values need to be set to 0 for averaging
        water_u = water_u.filled(0)
        water_v = water_v.filled(0)
        water_ang_rho = numpy.ma.masked_array(self.var_ang_rho, numpy.logical_not(self.var_mask_rho))
        water_lat_rho = numpy.ma.masked_array(self.var_lat_rho, numpy.logical_not(self.var_mask_rho))
        water_lon_rho = numpy.ma.masked_array(self.var_lon_rho, numpy.logical_not(self.var_mask_rho))

        u_rho, v_rho = average_uv2rho(water_u, water_v)

        rot_u_rho, rot_v_rho = rotate_uv2d(u_rho, v_rho, water_ang_rho)

        # u/v 1d arrays for interpolation
        u_compressed = numpy.ma.compressed(rot_u_rho)
        v_compressed = numpy.ma.compressed(rot_v_rho)
        lat_compressed = numpy.ma.compressed(water_lat_rho)
        lon_compressed = numpy.ma.compressed(water_lon_rho)

        return interp.interpolate_to_regular_grid((u_compressed, v_compressed),
                                                  lon_compressed, lat_compressed,
                                                  model_index.var_x, model_index.var_y,
                                                  interp_method=interp_method)

    def output_native_grid(self, time_index, target_depth):
        """Generate output using native grid coordinates"""

        u_target_depth, v_target_depth = vertical_interpolation(self.var_u, self.var_v, self.var_s_rho,
                                                                self.var_mask_rho, self.var_mask_u, self.var_mask_v,
                                                                self.var_zeta, self.var_h, self.var_hc, self.var_cs_r,
                                                                self.var_vtransform, self.num_eta, self.num_xi,
                                                                self.num_sigma, time_index, target_depth)

        # Create masked arrays to mask land values.
        water_u = numpy.ma.masked_array(u_target_depth, numpy.logical_not(self.var_mask_u))
        water_v = numpy.ma.masked_array(v_target_depth, numpy.logical_not(self.var_mask_v))

        # u/v masked values need to be set to 0 for averaging
        water_u = water_u.filled(0)
        water_v = water_v.filled(0)
        water_ang_rho = numpy.ma.masked_array(self.var_ang_rho, numpy.logical_not(self.var_mask_rho))
        water_lat_rho = numpy.ma.masked_array(self.var_lat_rho, numpy.logical_not(self.var_mask_rho))
        water_lon_rho = numpy.ma.masked_array(self.var_lon_rho, numpy.logical_not(self.var_mask_rho))

        # average and rotate u/v
        u_rho, v_rho = average_uv2rho(water_u, water_v)
        rot_u_rho, rot_v_rho = rotate_uv2d(u_rho, v_rho, water_ang_rho)

        # u/v 1d arrays for interpolation
        u_compressed = numpy.ma.compressed(rot_u_rho)
        v_compressed = numpy.ma.compressed(rot_v_rho)
        lat_compressed = numpy.ma.compressed(water_lat_rho)
        lon_compressed = numpy.ma.compressed(water_lon_rho)

        return u_compressed, v_compressed, lat_compressed, lon_compressed


def rotate_uv2d(u_rho, v_rho, water_ang_rho):
    """Rotate vectors by geometric angle.

    Args:
        u_rho: ``numpy.ma.masked_array`` containing u values averaged to rho.
        v_rho: ``numpy.ma.masked_array`` containing v values averaged to rho.
        water_ang_rho: ``numpy.ma.masked_array`` containing angle-of-rotation
            values for rho points.

    Returns:
        A 2-tuple containing rotated u and v arrays (in that order).
    """
    ang_sin = numpy.sin(water_ang_rho)
    ang_cos = numpy.cos(water_ang_rho)
    rot_u_rho = u_rho*ang_cos - v_rho*ang_sin
    rot_v_rho = u_rho*ang_sin + v_rho*ang_cos

    return rot_u_rho, rot_v_rho


def average_uv2rho(water_u, water_v):
    """Average u and v scalars to rho.

    U values at rho [eta,xi] are calculated by averaging u[eta,xi] with
    u[eta,xi+1], except for the final xi, where u at rho [eta,xi_max] is
    simply set to u[eta,xi_max-1].
    
    V values at rho [eta,xi] are calculated by averaging v[eta,xi] with
    u[eta+1,xi], except for the final eta, where v at rho [eta_max,xi] is
    simply set to v[eta_max-1,xi].
    
    Args:
        water_u: 2D numpy array containing U values to be averaged, with all
            NoData/Land-masked point values set to zero.
        water_v: 2D numpy array containing V values to be averaged, with all
            NoData/Land-masked point values set to zero.
    
    Returns:
        A 2-tuple of 2D ``numpy.ndarray``s containing u and v values
        (respectively) averaged to rho points.
    """
    num_eta = water_u.shape[0]
    num_xi = water_u.shape[1]

    # Average u values
    u_rho = numpy.ndarray([num_eta, num_xi])
    
    for eta in range(num_eta):
        for xi in range(num_xi-1):
            u_rho[eta, xi] = (water_u[eta, xi] + water_u[eta, xi+1])/2
        u_rho[eta, num_xi-1] = water_u[eta, num_xi-1]

    # Average v values
    v_rho = numpy.ndarray([num_eta, num_xi])
    
    for xi in range(num_xi):
        for eta in range(num_eta-1):
            v_rho[eta, xi] = (water_v[eta, xi] + water_v[eta+1, xi])/2
        v_rho[num_eta-1, xi] = water_v[num_eta-1, xi]

    return u_rho, v_rho


def vertical_interpolation(u, v, s_rho, mask_rho, mask_u, mask_v, zeta, h, hc, cs_r, vtransform, num_eta, num_xi, num_sigma, time_index, target_depth):
    """Vertically interpolate variables to target depth.

    Args:
        u: ``numpy.ndarray`` containing u values for entire grid.
        v: ``numpy.ndarray`` containing v values for entire grid.
        s_rho: ``numpy.ndarray`` s-coordinate at rho points, -1 min 0 max,
            positive up.
        zeta: ``numpy.ndarray`` containing MSL free surface at rho points in
            meters.
        h: ``numpy.ndarray`` containing bathymetry at rho points.
        hc: ``numpy.ndarray`` containing s-coordinate parameter, critical
            depth.
        cs_r: ``numpy.ndarray`` containing s-coordinate stretching curves at
            rho points.
        mask_u: ``numpy.ndarray`` containing mask values for u points.
        mask_v: ``numpy.ndarray`` containing mask values for v points.
        mask_rho: ``numpy.ndarray`` containing mask values for rho points.
        vtransform: Vertical terrain-following transformation equation.
        num_eta: eta dimensions.
        num_xi: xi dimensions.
        num_sigma: Number of sigma layers.
        time_index: Single forecast time index value.
        target_depth: The target depth-below-sea-surface to which water
            currents will be interpolated, in meters. Must be zero or greater.
            For areas shallower than double this value, values will be
            interpolated to half the water column height instead. For
            navigationally significant currents, a value of 4.5 is recommended.

    Returns:
        A 2-tuple containing u and v (in that order) value arrays calculated at
        target depth.
    """
    zeta = numpy.ma.masked_array(zeta[time_index, :, :], numpy.logical_not(mask_rho))
    true_depth = h + zeta
    z = numpy.ma.empty(shape=[num_sigma, num_eta, num_xi])

    if vtransform == 1:
        # Roms vertical transformation equation 1
        for k in range(num_sigma):
            s = ((hc * s_rho[k]) + (h - hc) * cs_r[k])
            z[k, :, :] = s + zeta * (1 + s/h)
    else:
        # Roms vertical transformation equation 2 GOMOFS Only
        for k in range(num_sigma):
            s = ((hc * s_rho[k]) + (h*cs_r[k]))/(h + hc)
            z[k, :, :] = zeta + ([zeta + h]*s)

    if target_depth < 0:
        raise Exception('Target depth must be positive')

    # Convert target depth-below-surface to negative value, since sigma is
    # negative (0 =~ surface, -1 =~ seafloor) [even if sigma values are stored
    # in the NetCDF as positive-down, the values are converted to negative
    # automatically in init_handles()]
    # For areas shallower than the target depth, depth is half the total depth
    interp_depth = -1 * numpy.minimum(target_depth * 2, true_depth) / 2

    u_target_depth = numpy.ma.empty(shape=[num_eta, num_xi])
    v_target_depth = numpy.ma.empty(shape=[num_eta, num_xi])

    # Perform vertical linear interpolation on u/v values to target depth
    for eta in range(num_eta):
        for xi in range(num_xi):
            if not zeta.mask[eta, xi]:
                if mask_u[eta, xi] != 0:
                    u_interp_depth = interpolate.interp1d(z[:, eta, xi], u[time_index, :, eta, xi], fill_value='extrapolate')
                    u_target_depth[eta, xi] = u_interp_depth(interp_depth.data[eta, xi])

                if mask_v[eta, xi] != 0:
                    v_interp_depth = interpolate.interp1d(z[:, eta, xi], v[time_index, :, eta, xi], fill_value='extrapolate')
                    v_target_depth[eta, xi] = v_interp_depth(interp_depth.data[eta, xi])

    return u_target_depth, v_target_depth
