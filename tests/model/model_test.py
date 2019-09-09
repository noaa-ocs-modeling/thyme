from collections import namedtuple

import numpy
import pytest

from thyme.model.model import regular_uv_to_speed_direction

RegGridValues = namedtuple(
    'RegGridValues',
    ['u',
     'v',
     'expected_speed',
     'expected_direction'])

@pytest.fixture
def reg_grid_values():
    u = numpy.ma.masked_array(
        [
            [0, 0.3, 0.5],
            [0.3, 0.5, 0.7],
            [999, 999, 999]
        ], mask = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]
        ])

    v = numpy.ma.masked_array(
        [
            [0, 0.1, 0.2],
            [0.4, 0.3, 999],
            [999, 999, 0.7]
        ], mask = [
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 0]
        ])

    combined_mask = [
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1]
        ]

    expected_speed = numpy.ma.masked_array(
        [
            [0.0, 0.6146974563598633, 1.0467920303344727],
            [0.971921980381012, 1.1334460973739624, 999],
            [999, 999, 999]
        ], mask=combined_mask)

    expected_direction = numpy.ma.masked_array(
        [
            [90.0, 71.56504821777344, 68.19859313964844],
            [36.869895935058594, 59.0362434387207, 999],
            [999, 999, 999]
        ], mask=combined_mask)

    return RegGridValues(u, v, expected_speed, expected_direction)


def test_regular_uv_to_speed_direction(reg_grid_values):
    """Test conversion from u/v (m/s) to speed/direction (knots/degrees)"""
    speed, direction = regular_uv_to_speed_direction(reg_grid_values.u, reg_grid_values.v)
    print(f"speed: {speed}")
    print(f"direction: {direction}")
    assert numpy.allclose(speed, reg_grid_values.expected_speed)
    assert numpy.allclose(direction, reg_grid_values.expected_direction)

