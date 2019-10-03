from collections import namedtuple

import numpy
import pytest

from thyme.util import interp


# Container for test fixture's expected input/output values
InterpValues = namedtuple('InterpValues', ['values_in', 'x_in', 'y_in', 'x_out', 'y_out', 'expected_values_out'])


@pytest.fixture
def interp_values():
    lon_in = numpy.array([-73.6 + i*0.2 for i in range(3)])
    lat_in = numpy.array([42.2 + i*0.2 for i in range(3)])
    x_in, y_in = [result.flatten() for result in numpy.meshgrid(lon_in, lat_in)]
    x_out = numpy.array([-74 + i*0.1 for i in range(11)])
    y_out = numpy.array([42 + i*0.1 for i in range(11)])
    values_in = [
        numpy.array([0, 20.0, 40.0, 20.0, 50.0, 80.0, 40.0, 80.0, 100.0])
    ]
    expected_values_out = [
        numpy.ma.masked_array([
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0.0, 10.0, 20.0, 30.0, 40.0, 0, 0),
            (0, 0, 0, 0, 10.0, 20.0, 35.0, 50.0, 60.0, 0, 0),
            (0, 0, 0, 0, 20.0, 35.0, 50.0, 65.0, 80.0, 0, 0),
            (0, 0, 0, 0, 30.0, 50.0, 65.0, 80.0, 90.0, 0, 0),
            (0, 0, 0, 0, 40.0, 60.0, 80.0, 90.0, 100.0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        ], mask=[
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        ])
    ]
    return InterpValues(values_in, x_in, y_in, x_out, y_out, expected_values_out)


# Commenting out GDAL interp test until swig issue is fixed
# See https://github.com/OSGeo/gdal/issues/1677
#
# def test_gdal_interp(interp_values):
#    """Test gdal interpolation to regular grid"""
#    print("interp_values: " + repr(interp_values))
#    values_out = interp.gdal_interpolate_to_regular_grid(
#        interp_values.values_in,
#        interp_values.x_in,
#        interp_values.y_in,
#        interp_values.x_out,
#        interp_values.y_out)
#   values_out_masked = [numpy.ma.masked_array(x, mask=interp_values.expected_
#    values_out[i].mask) for i, x in enumerate(values_out)]
#
#    print("values_out_masked: " + repr(values_out_masked))
#
#    assert all([numpy.allclose(values_out_masked[i], interp_values.expected_
#    values_out[i]) for i in range(len(values_out_masked))])


def test_scipy_interp(interp_values):
    """Test scipy interpolation to regular grid"""
    print("interp_values: " + repr(interp_values))
    values_out = interp.scipy_interpolate_to_regular_grid(
        interp_values.values_in,
        interp_values.x_in,
        interp_values.y_in,
        interp_values.x_out,
        interp_values.y_out)
    values_out_masked = [numpy.ma.masked_array(x, mask=interp_values.expected_values_out[i].mask) for i, x in enumerate(values_out)]

    print("values_out_masked: " + repr(values_out_masked))

    assert all([numpy.allclose(values_out_masked[i], interp_values.expected_values_out[i]) for i in range(len(values_out_masked))])


def test_generic_interp(interp_values):
    """Test generic interpolation function"""
    print("interp_values: " + repr(interp_values))
    values_out = interp.interpolate_to_regular_grid(
        interp_values.values_in,
        interp_values.x_in,
        interp_values.y_in,
        interp_values.x_out,
        interp_values.y_out,
        interp_method=interp.INTERP_METHOD_SCIPY)
    values_out_masked = [numpy.ma.masked_array(x, mask=interp_values.expected_values_out[i].mask) for i, x in enumerate(values_out)]

    print("values_out_masked: " + repr(values_out_masked))

    assert all([numpy.allclose(values_out_masked[i], interp_values.expected_values_out[i]) for i in range(len(values_out_masked))])

