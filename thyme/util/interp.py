"""
Spatial interpolation functionality.
"""
import numpy
from osgeo import gdal, ogr, osr
from scipy.interpolate import griddata


# Default method for horizontal interpolation
INTERP_METHOD_SCIPY = 'scipy'

# Alternative method for horizontal interpolation
INTERP_METHOD_GDAL = 'gdal'


def interpolate_to_regular_grid(values, x_in, y_in, x_out, y_out, interp_method=INTERP_METHOD_SCIPY):
    """Linear-interpolate irregularly-spaced values to regular grid.

    Args:
        values: `tuple` or `list` of `numpy.ma.masked_array`s containing values
            to be interpolated to the output grid.
        x_in: `numpy.ndarray` containing x-position of each input value. Length
            of this array must match the length of each array in `values`.
        y_in: `numpy.ndarray` containing y-position of each input value. Length
            of this array must match the length of each array in `values`.
        x_out: `numpy.ndarray` containing x-positions of output grid columns.
        y_out: `numpy.ndarray` containing y-positions of output grid rows.
        interp_method: The desired interpolation method. Defaults to
            `INTERP_METHOD_SCIPY`.

    Returns:
        A `tuple` containing the interpolated values for each input array,
        corresponding with the order specified in `values`.
    """
    if interp_method == INTERP_METHOD_SCIPY:
        return scipy_interpolate_to_regular_grid(values, x_in, y_in, x_out, y_out)
    elif interp_method == INTERP_METHOD_GDAL:
        return gdal_interpolate_to_regular_grid(values, x_in, y_in, x_out, y_out)

    raise ValueError(f"Invalid interpolation method specified [{interp_method}]. Supported values: {INTERP_METHOD_SCIPY}, {INTERP_METHOD_GDAL}")


def scipy_interpolate_to_regular_grid(values, x_in, y_in, x_out, y_out):
    """Linear-interpolate irregularly-spaced values to regular grid.

    Uses `scipy.interpolate.griddata` for linear interpolation.

    Args:
        values: `tuple` or `list` of `numpy.ma.masked_array`s containing values
            to be interpolated to the output grid.
        x_in: `numpy.ndarray` containing x-position of each input value. Length
            of this array must match the length of each array in `values`.
        y_in: `numpy.ndarray` containing y-position of each input value. Length
            of this array must match the length of each array in `values`.
        x_out: `numpy.ndarray` containing x-positions of output grid columns.
        y_out: `numpy.ndarray` containing y-positions of output grid rows.

    Returns:
        A `tuple` containing the interpolated values for each input array,
        corresponding with the order specified in `values`.
    """
    x, y = numpy.meshgrid(x_out, y_out)
    coords = numpy.column_stack((x_in, y_in))
    calc_values = []
    for value_array in values:
        calc_values.append(griddata(coords, value_array, (x, y), method='linear'))

    return tuple(calc_values)


def gdal_interpolate_to_regular_grid(values, x_in, y_in, x_out, y_out):
    """Linear-interpolate irregularly-spaced values to regular grid.

    Uses `gdal.Grid` for linear interpolation.

    NOTE: as of 9/2019, this GDAL interpolation is not working, seemingly
    because of a bug with swig: https://github.com/OSGeo/gdal/issues/1677

    Args:
        values: `tuple` or `list` of `numpy.ma.masked_array`s containing values
            to be interpolated to the output grid.
        x_in: `numpy.ndarray` containing x-position of each input value. Length
            of this array must match the length of each array in `values`.
        y_in: `numpy.ndarray` containing y-position of each input value. Length
            of this array must match the length of each array in `values`.
        x_out: `numpy.ndarray` containing x-positions of output grid columns.
        y_out: `numpy.ndarray` containing y-positions of output grid rows.

    Returns:
        A `tuple` containing the interpolated values for each input array,
        corresponding with the order specified in `values`.
    """
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    ds = gdal.GetDriverByName('Memory').Create('', 0, 0, 0, gdal.GDT_Float32)
    layer = ds.CreateLayer('irregular_points', srs=srs, geom_type=ogr.wkbPoint)

    def field_name(x):
        return 'field_{}'.format(x)

    for n in range(len(values)):
        layer.CreateField(ogr.FieldDefn(field_name(n + 1), ogr.OFTReal))

    for i in range(len(x_in)):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x_in[i], y_in[i])
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(point)
        for n in range(len(values)):
            layer.SetField(field_name(n + 1), values[n][i])
        layer.CreateFeature(feature)

    calc_values = []
    for n in range(len(values)):
        calc_values.append(
            gdal.Grid('.tif'.format(field_name(n + 1)), ds, format='MEM',
                      width=len(x_out), height=len(y_out),
                      algorithm='linear:nodata=0.0', zfield=field_name(n + 1)).ReadAsArray())

    return tuple(calc_values)
