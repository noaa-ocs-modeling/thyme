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

def interpolate_uv_to_regular_grid(u, v, lat, lon, model_index, interp_method=INTERP_METHOD_SCIPY):
    """Linear-interpolate irregularly-spaced u/v values to regular grid.

    Args:
        u: `numpy.ma.masked_array` containing u values with NoData/land values
            masked out.
        v: `numpy.ma.masked_array` containing v values with NoData/land values
            masked out.
        lat: `numpy.ndarray` containing latitude positions of corresponding
            input u/v values.
        lon: `numpy.ndarray` containing longitude positions of corresponding
            input u/v values
        model_index: `ModelIndexFile` containing output regular grid
            definition.
        interp_method: The desired interpolation method. Defaults to
            `INTERP_METHOD_SCIPY`.
    """
    if interp_method == INTERP_METHOD_SCIPY:
        return scipy_interpolate_uv_to_regular_grid(u, v, lat, lon, model_index)
    elif interp_method == INTERP_METHOD_GDAL:
        return gdal_interpolate_uv_to_regular_grid(u, v, lat, lon, model_index)

    raise ValueError(f"Invalid interpolation method specified [{interp_method}]. Supported values: {INTERP_METHOD_SCIPY}, {INTERP_METHOD_GDAL}")

def scipy_interpolate_uv_to_regular_grid(u, v, lat, lon, model_index):
    """Linear-interpolate irregularly-spaced u/v values to regular grid.
    
    Uses `scipy.interpolate.griddata` for linear interpolation.

    Args:
        u: `numpy.ma.masked_array` containing u values with NoData/land values
            masked out.
        v: `numpy.ma.masked_array` containing v values with NoData/land values
            masked out.
        lat: `numpy.ndarray` containing latitude positions of corresponding
            input u/v values.
        lon: `numpy.ndarray` containing longitude positions of corresponding
            input u/v values
        model_index: `ModelIndexFile` containing output regular grid
            definition.
    """
    x, y = numpy.meshgrid(model_index.var_x, model_index.var_y)
    coords = numpy.column_stack((lon, lat))
    reg_grid_u = griddata(coords, u, (x, y), method='linear')
    reg_grid_v = griddata(coords, v, (x, y), method='linear')

    return reg_grid_u, reg_grid_v


def gdal_interpolate_uv_to_regular_grid(u, v, lat, lon, model_index):
    """Linear-interpolate irregularly-spaced u/v values to regular grid.

    Uses `gdal.Grid` for linear interpolation.

    Args:
        u: `numpy.ma.masked_array` containing u values with NoData/land values
            masked out.
        v: `numpy.ma.masked_array` containing v values with NoData/land values
            masked out.
        lat: `numpy.ndarray` containing latitude positions of corresponding
            input u/v values.
        lon: `numpy.ndarray` containing longitude positions of corresponding
            input u/v values
        model_index: `ModelIndexFile` containing output regular grid
            definition.
    """
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    ds = gdal.GetDriverByName('Memory').Create('', 0, 0, 0, gdal.GDT_Float32)
    layer = ds.CreateLayer('irregular_points', srs=srs, geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn('u', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('v', ogr.OFTReal))

    for i in range(len(v)):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon[i], lat[i])
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(point)
        feature.SetField('u', u[i])
        feature.SetField('v', v[i])
        layer.CreateFeature(feature)

    # Input ogr object to gdal grid and interpolate irregularly spaced u/v
    # to a regular grid
    dst_u = gdal.Grid('u.tif', ds, format='MEM', width=model_index.dim_x.size, height=model_index.dim_y.size,
                      algorithm='linear:nodata=0.0', zfield='u')
    dst_v = gdal.Grid('v.tif', ds, format='MEM', width=model_index.dim_x.size, height=model_index.dim_y.size,
                      algorithm='linear:nodata=0.0', zfield='v')

    reg_grid_u = dst_u.ReadAsArray()
    reg_grid_v = dst_v.ReadAsArray()

    return reg_grid_u, reg_grid_v
