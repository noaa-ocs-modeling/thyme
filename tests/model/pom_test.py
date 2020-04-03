from collections import namedtuple

import numpy
import pytest

from thyme.model.pom import vertical_interpolation

VerticalValues = namedtuple(
    'VerticalValues',
    ['u',
     'v',
     'mask',
     'zeta',
     'depth',
     'sigma',
     'num_sigma',
     'num_nx',
     'num_ny',
     'time_index',
     'target_depth_surface',
     'target_depth_default',
     'target_depth_deep',
     'expected_u_target_depth_default',
     'expected_v_target_depth_default',
     'expected_u_target_depth_surface',
     'expected_v_target_depth_surface',
     'expected_u_target_depth_deep',
     'expected_v_target_depth_deep'])


@pytest.fixture
def vertical_values():

    time_index = 0
    num_nx = 3
    num_ny = 3
    num_sigma = 7

    target_depth_default = 4.5
    target_depth_surface = 0
    target_depth_deep = 12

    u = numpy.array(
        [[
            [[-0.14429612457752228, -0.1557592898607254, -0.1639900505542755],
             [-0.13332930207252502, -0.13850903511047363, -0.14700627326965332],
             [-0.08938728272914886, -0.10011885315179825, -0.10417534410953522]],

            [[-0.17176321148872375, -0.18649248778820038, -0.1963706910610199],
             [-0.15961810946464539, -0.16394512355327606, -0.17563748359680176],
             [-0.10463598370552063, -0.11853943765163422, -0.12382897734642029]],

            [[-0.1860588788986206, -0.2024313062429428, -0.21306045353412628],
             [-0.17317943274974823, -0.1772005558013916, -0.19041943550109863],
             [-0.11270792037248611, -0.1282253861427307, -0.13397538661956787]],

            [[-0.1959017515182495, -0.21337155997753143, -0.22445325553417206],
             [-0.1824016571044922, -0.18625672161579132, -0.20042560994625092],
             [-0.11817137151956558, -0.13474631309509277, -0.14073477685451508]],

            [[-0.20371614396572113, -0.222026988863945, -0.23342110216617584],
             [-0.18960487842559814, -0.19335819780826569, -0.2081872522830963],
             [-0.12233472615480423, -0.13967420160770416, -0.1458158940076828]],

            [[-0.21044501662254333, -0.22945916652679443, -0.2410876452922821],
             [-0.1957058608531952, -0.19939996302127838, -0.2147126942873001],
             [-0.12573498487472534, -0.14364899694919586, -0.1499125063419342]],

            [[-0.21648438274860382, -0.23612338304519653, -0.2479390650987625],
             [-0.2011163979768753, -0.20478609204292297, -0.22046254575252533],
             [-0.12864403426647186, -0.14699482917785645, -0.1533738672733307]]

        ]])

    v = numpy.array(
        [[
            [[-0.015962883830070496, -0.014421091414988041, -0.015121867880225182],
             [-0.013614140450954437, -0.005653033033013344, -0.0029921410605311394],
             [-0.002787810517475009, -0.0023808081168681383, 0.006570389028638601]],

            [[-0.01818399503827095, -0.016664784401655197, -0.016933994367718697],
             [-0.015678774565458298, -0.0062980614602565765, -0.0029219179414212704],
             [-0.0032500678207725286, -0.002716263523325324, 0.00792204961180687]],

            [[-0.01906922645866871, -0.017558123916387558, -0.017497723922133446],
             [-0.016553614288568497, -0.006449156906455755, -0.002613181946799159],
             [-0.0034973544534295797, -0.002890110481530428, 0.008606006391346455]],

            [[-0.019496159628033638, -0.017993737012147903, -0.01763434149324894],
             [-0.017030581831932068, -0.006439469289034605, -0.0022337716072797775],
             [-0.0036818813532590866, -0.003022562013939023, 0.009041817858815193]],

            [[-0.01969190314412117, -0.018201317638158798, -0.017547737807035446],
             [-0.017312992364168167, -0.006345934234559536, -0.0018080766312777996],
             [-0.0038449172861874104, -0.003145989030599594, 0.009346261620521545]],

            [[-0.019737228751182556, -0.018264643847942352, -0.0173049159348011],
             [-0.01747279241681099, -0.006191743537783623, -0.0013355917762964964],
             [-0.004004715010523796, -0.003275529947131872, 0.009564205072820187]],

            [[-0.019665192812681198, -0.018219994381070137, -0.01693047396838665],
             [-0.01753874495625496, -0.005984888412058353, -0.0008128402987495065],
             [-0.004172911401838064, -0.0034219594672322273, 0.009713913314044476]]

        ]])

    mask = numpy.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

    zeta = numpy.array(
        [[
             [0.6813687682151794, 0.6795873641967773, 0.6781861782073975],
             [0.6826923489570618, 0.6809894442558289, 0.6800354719161987],
             [-0.31109474877147, -0.310917392324487, -0.310739945656742]
        ]])

    depth = numpy.array(
        [
            [2, 3.527, 4.8061],
            [4.2, 4.5, 0.2],
            [5.5952, 12.5469, 7.042]
        ])

    sigma = numpy.array(
        [
            0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1
        ])

    expected_u_target_depth_default = numpy.array(
        [
            [-0.14429612, -0.15575929, -0.16399005],
            [-0.1333293, -0.13850904, -0.14700627],
            [-0.08938728, -0.10011885, -0.10417534]
        ])
    expected_v_target_depth_default = numpy.array(
        [
            [-0.01596288, -0.01442109, -0.01512187],
            [-0.01361414, -0.00565303, -0.00299214],
            [-0.00278781, -0.00238081,  0.00657039]
        ])

    expected_u_target_depth_surface = numpy.array(
        [
            [-0.06189488, -0.06355971, -0.06684815],
            [-0.0544629, -0.06220079, -0.06111266],
            [-0.04364119, -0.05947188, -0.04521446]
        ])

    expected_v_target_depth_surface = numpy.array(
         [
             [-0.00929955, -0.00769001, -0.00968549],
             [-0.00742024, -0.00371795, -0.00320281],
             [-0.00140104, -0.00164059,  0.00251541]
         ])

    expected_u_target_depth_deep = numpy.array(
        [
            [-0.06189488, -0.06355971, -0.06684815],
            [-0.0544629, -0.06220079, -0.06111266],
            [-0.04364119, -0.04485711, -0.04521446]
        ])

    expected_v_target_depth_deep = numpy.array(
        [
            [-0.00929955, -0.00769001, -0.00968549],
            [-0.00742024, -0.00371795, -0.00320281],
            [-0.00140104, -0.00137444,  0.00251541]
        ])

    return VerticalValues(u, v, mask, zeta, depth, sigma, num_sigma, num_ny, num_nx, time_index, target_depth_default,
                          target_depth_surface, target_depth_deep, expected_u_target_depth_default,
                          expected_v_target_depth_default, expected_u_target_depth_surface,
                          expected_v_target_depth_surface, expected_u_target_depth_deep, expected_v_target_depth_deep)


def test_vertical_interpolation(vertical_values):
    """Test vertical interpolation of u/v to default target depth."""
    u_target_depth, v_target_depth = vertical_interpolation(vertical_values.u, vertical_values.v, vertical_values.mask,
                                                            vertical_values.zeta, vertical_values.depth,
                                                            vertical_values.sigma, vertical_values.num_sigma,
                                                            vertical_values.num_ny, vertical_values.num_nx,
                                                            vertical_values.time_index,
                                                            vertical_values.target_depth_default)

    # print(f"u_target_depth_default: {u_target_depth}")
    # print(f"v_target_depth_default: {v_target_depth}")
    assert numpy.allclose(u_target_depth, vertical_values.expected_u_target_depth_default)
    assert numpy.allclose(v_target_depth, vertical_values.expected_v_target_depth_default)


def test_vertical_interpolation_at_surface(vertical_values):
    """Test vertical interpolation of u/v target depth at surface."""
    u_target_depth, v_target_depth = vertical_interpolation(vertical_values.u, vertical_values.v, vertical_values.mask,
                                                            vertical_values.zeta, vertical_values.depth,
                                                            vertical_values.sigma, vertical_values.num_sigma,
                                                            vertical_values.num_ny, vertical_values.num_nx,
                                                            vertical_values.time_index,
                                                            vertical_values.target_depth_surface)

    # print(f"u_target_depth_surface: {u_target_depth}")
    # print(f"v_target_depth_surface: {v_target_depth}")
    assert numpy.allclose(u_target_depth, vertical_values.expected_u_target_depth_surface)
    assert numpy.allclose(v_target_depth, vertical_values.expected_v_target_depth_surface)


def test_vertical_interpolation_deep(vertical_values):
    """Test vertical interpolation of u/v to deeper target depth."""
    u_target_depth, v_target_depth = vertical_interpolation(vertical_values.u, vertical_values.v, vertical_values.mask,
                                                            vertical_values.zeta, vertical_values.depth,
                                                            vertical_values.sigma, vertical_values.num_sigma,
                                                            vertical_values.num_ny, vertical_values.num_nx,
                                                            vertical_values.time_index,
                                                            vertical_values.target_depth_deep)

    # print(f"u_target_depth_deep: {u_target_depth}")
    # print(f"v_target_depth_deep: {v_target_depth}")
    assert numpy.allclose(u_target_depth, vertical_values.expected_u_target_depth_deep)
    assert numpy.allclose(v_target_depth, vertical_values.expected_v_target_depth_deep)
