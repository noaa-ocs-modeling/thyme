from collections import namedtuple

import numpy
import pytest

from thyme.model.roms import average_uv2rho
from thyme.model.roms import rotate_uv2d


UVAverageValues = namedtuple(
    'UVAverageValues',
    ['u',
     'v',
     'expected_averaged_u',
     'expected_averaged_v'])


@pytest.fixture
def uv_average_values():
    # U is averaged from left to right within inner arrays (xi dimension)
    # except for final records which are copied
    # e.g.:
    #   [[val1, val2, val3], [val4, val5, val6]]
    #     is averaged to:
    #   [[(val1 + val2)/2, (val2 + val3)/2, val3], [(val4 + val5)/2, (val5 + val6)/2, val6]]
    u = numpy.array(
        [
            [-0.6536411, -0.5898356, -0.5823435],
            [-0.730157, -0.7684499, -0.7612157],
            [-0.6766191, -0.6804294, -0.6829212]

        ])

    # V is averaged from top to bottom across outer array (eta dimension),
    # except for final records which are copied
    # e.g.:
    #   [[val1, val2, val3], [val4, val5, val6]]
    #     is averaged to:
    #   [[(val1 + val4)/2, (val2 + val5)/2, (val3 + val6)/2], [val4, val5, val6]]
    v = numpy.array(
        [
            [-0.01937735, -0.0215912, -0.02442237],
            [-0.06249624, -0.06321328, -0.04191912],
            [0.05146192, 0.0615576, 0.06413107]

        ])

    expected_averaged_u = numpy.array(
        [
            [-0.62173835, -0.58608955, -0.5823435],
            [-0.74930345, -0.7648328, -0.7612157],
            [-0.67852425, -0.6816753, -0.6829212]

        ])

    expected_averaged_v = numpy.array(
        [
            [-0.04093679, -0.04240224, -0.03317075],
            [-0.00551716, -0.00082784, 0.01110598],
            [0.05146192, 0.0615576,  0.06413107]

        ])

    return UVAverageValues(u, v, expected_averaged_u, expected_averaged_v)


def test_average_uv2rho(uv_average_values):
    """Test averaging u/v values to rho points"""
    u_rho, v_rho = average_uv2rho(uv_average_values.u, uv_average_values.v)
    print(f"u_rho: {u_rho}")
    print(f"v_rho: {v_rho}")
    assert numpy.allclose(u_rho, uv_average_values.expected_averaged_u)
    assert numpy.allclose(v_rho, uv_average_values.expected_averaged_v)
