from collections import namedtuple

import numpy
from numpy.testing import assert_approx_equal
import pytest

from thyme.grid.regulargrid import RegularGrid

RegularGridValues = namedtuple(
    'RegularGridValues',
    ['lon_min',
     'lat_min',
     'lon_max',
     'lat_max',
     'target_cellsize_meters',
     'expected_cellsize_x',
     'expected_cellsize_y',
     'expected_x_coords',
     'expected_y_coords'])


@pytest.fixture
def reg_grid_values():
    lon_min = -73.1
    lat_min = 42.0
    lon_max = -73.0
    lat_max = 42.1
    target_cellsize_meters = 500
    expected_cellsize_x = 0.005882352941176137
    expected_cellsize_y = 0.005882352941176137
    expected_x_coords = [-73.09705882352941, -73.09117647058824, -73.08529411764707, -73.0794117647059, -73.07352941176472, -73.06764705882355, -73.06176470588238, -73.05588235294121, -73.05000000000004, -73.04411764705887, -73.0382352941177, -73.03235294117653, -73.02647058823536, -73.02058823529418, -73.01470588235301, -73.00882352941184, -73.00294117647067]
    expected_y_coords = [42.002941176470586, 42.008823529411764, 42.01470588235294, 42.02058823529412, 42.0264705882353, 42.03235294117648, 42.038235294117655, 42.04411764705883, 42.05000000000001, 42.05588235294119, 42.06176470588237, 42.067647058823546, 42.073529411764724, 42.0794117647059, 42.08529411764708, 42.09117647058826, 42.09705882352944]

    return RegularGridValues(lon_min, lat_min, lon_max, lat_max, target_cellsize_meters, expected_cellsize_x, expected_cellsize_y, expected_x_coords, expected_y_coords)


def test_calc_cellsizes(reg_grid_values):
    """Test cell size calculation"""
    cellsize_x, cellsize_y = RegularGrid.calc_cellsizes(
        reg_grid_values.lon_min,
        reg_grid_values.lat_min,
        reg_grid_values.lon_max,
        reg_grid_values.lat_max,
        reg_grid_values.target_cellsize_meters)
    print(f"Calculated cellsize_x: {cellsize_x}")
    print(f"Calculated cellsize_y: {cellsize_y}")

    assert_approx_equal(cellsize_x, reg_grid_values.expected_cellsize_x)
    assert_approx_equal(cellsize_y, reg_grid_values.expected_cellsize_y)

def test_calc_gridpoints(reg_grid_values):
    """Test grid point calculation"""
    reg_grid = RegularGrid(
        reg_grid_values.lon_min,
        reg_grid_values.lat_min,
        reg_grid_values.lon_max,
        reg_grid_values.lat_max,
        reg_grid_values.expected_cellsize_x,
        reg_grid_values.expected_cellsize_y)
    print(f"Calculated x_coords: {reg_grid.x_coords}")
    print(f"Calculated y_coords: {reg_grid.y_coords}")
    assert numpy.allclose(reg_grid.x_coords, reg_grid_values.expected_x_coords)
    assert numpy.allclose(reg_grid.x_coords, reg_grid_values.expected_x_coords)
