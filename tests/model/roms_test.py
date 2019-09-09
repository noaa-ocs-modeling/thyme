from collections import namedtuple

import numpy
import pytest

from thyme.model.roms import average_uv2rho


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
            [0, 0.3, 0.5],
            [0.3, 0.5, 0.7],
            [-0.2, 0.1, -0.7]
        ])

    # V is averaged from top to bottom across outer array (eta dimension),
    # except for final records which are copied
    # e.g.:
    #   [[val1, val2, val3], [val4, val5, val6]]
    #     is averaged to:
    #   [[(val1 + val4)/2, (val2 + val5)/2, (val3 + val6)/2], [val4, val5, val6]]
    v = numpy.array(
        [
            [0, 0.1, 0.2],
            [0.4, 0.3, 0.5],
            [-0.5, 0.2, 0.7]
        ])

    expected_averaged_u = numpy.array(
        [
            [0.15, 0.4, 0.5],
            [0.4, 0.6, 0.7],
            [-0.05, -0.3, -0.7]
        ])

    expected_averaged_v = numpy.array(
        [
            [0.2, 0.2, 0.35],
            [-0.05, 0.25, 0.6],
            [-0.5, 0.2, 0.7]
        ])

    return UVAverageValues(u, v, expected_averaged_u, expected_averaged_v)


def test_average_uv2rho(uv_average_values):
    """Test averaging u/v values to rho points"""
    u_rho, v_rho = average_uv2rho(uv_average_values.u, uv_average_values.v)
    print(f"u_rho: {u_rho}")
    print(f"v_rho: {v_rho}")
    assert numpy.allclose(u_rho, uv_average_values.expected_averaged_u)
    assert numpy.allclose(v_rho, uv_average_values.expected_averaged_v)

