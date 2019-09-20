from collections import namedtuple

import numpy
import pytest

from thyme.model.model import regular_uv_to_speed_direction
from thyme.model.model import irregular_uv_to_speed_direction

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
        ], mask=[
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]
        ])

    v = numpy.ma.masked_array(
        [
            [0, 0.1, 0.2],
            [0.4, 0.3, 999],
            [999, 999, 0.7]
        ], mask=[
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


IrregularGridValues = namedtuple(
    'RegGridValues',
    ['u',
     'v',
     'expected_speed',
     'expected_direction'])


@pytest.fixture
def irregular_grid_values():
    u = numpy.ma.masked_array(
        [
            [-0.145427],
            [-0.148755],
            [-0.150399],
            [-0.154676],
            [-0.119714],
            [999]
        ], mask=[
            [0],
            [0],
            [0],
            [0],
            [0],
            [1]
        ])

    v = numpy.ma.masked_array(
        [
            [999],
            [0.039037],
            [0.019768],
            [0.008497],
            [0.026284],
            [0.040731]
        ], mask=[
            [1],
            [0],
            [0],
            [0],
            [0],
            [0]
        ])

    combined_mask = [
            [1],
            [0],
            [0],
            [0],
            [0],
            [1]
        ]

    expected_speed = numpy.ma.masked_array(
        [
            [999],
            [0.29894739389419556],
            [0.294866681098938],
            [0.301119327545166],
            [0.23824812471866608],
            [999]
        ], mask=combined_mask)

    expected_direction = numpy.ma.masked_array(
        [
            [999],
            [284.70428466796875],
            [277.48785400390625],
            [273.14434814453125],
            [282.3831787109375],
            [999]
        ], mask=combined_mask)

    return IrregularGridValues(u, v, expected_speed, expected_direction)


def test_irregular_uv_to_speed_direction(irregular_grid_values):
    """Test conversion from u/v (m/s) to speed/direction (knots/degrees)"""
    speed, direction = irregular_uv_to_speed_direction(irregular_grid_values.u, irregular_grid_values.v)
    print(f"speed: {speed}")
    print(f"direction: {direction}")
    assert numpy.allclose(speed, irregular_grid_values.expected_speed)
    assert numpy.allclose(direction, irregular_grid_values.expected_direction)
