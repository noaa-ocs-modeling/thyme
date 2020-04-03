from collections import namedtuple

import numpy
import pytest

from thyme.model.roms import average_uv2rho
from thyme.model.roms import rotate_uv2d
from thyme.model.roms import vertical_interpolation

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


UVRotateValues = namedtuple(
    'UVRotateValues',
    ['u_rho',
     'v_rho',
     'ang_rho',
     'expected_rotated_u',
     'expected_rotated_v'])


@pytest.fixture
def uv_rotate_values():
    u_rho = numpy.array(
        [
            [-0.62173835, -0.58608955, -0.5823435],
            [-0.74930345, -0.7648328, -0.7612157],
            [-0.67852425, -0.6816753, -0.6829212]
        ])

    v_rho = numpy.array(
        [
            [-0.04093679, -0.04240224, -0.03317075],
            [-0.00551716, -0.00082784, 0.01110598],
            [0.05146192, 0.0615576,  0.06413107]
        ])
    ang_rho = numpy.array(
        [
            [-0.312156994200958, -0.311980178561144, -0.311803272848723],
            [-0.311626277038169, -0.311449191104021, -0.311272015022893],
            [-0.31109474877147, -0.310917392324487, -0.310739945656742]
        ])

    expected_rotated_u = numpy.array(
        [
            [-0.604264, -0.57081273, -0.56443999],
            [-0.71490575, -0.72829074, -0.72123387],
            [-0.63020189, -0.63015878, -0.63060553]

        ])

    expected_rotated_v = numpy.array(
        [
            [0.15198492, 0.13954119, 0.14707737],
            [0.22449024, 0.23358615, 0.24370964],
            [0.25668866, 0.26715252, 0.26987187]

        ])

    return UVRotateValues(u_rho, v_rho, ang_rho, expected_rotated_u, expected_rotated_v)


def test_rotate_uv2d(uv_rotate_values):
    """Test rotating u/v values by geometric angle"""
    rot_u_rho, rot_v_rho = rotate_uv2d(uv_rotate_values.u_rho, uv_rotate_values.v_rho, uv_rotate_values.ang_rho)
    print(f"rot_u: {rot_u_rho}")
    print(f"rot_v: {rot_v_rho}")
    assert numpy.allclose(rot_u_rho, uv_rotate_values.expected_rotated_u)
    assert numpy.allclose(rot_v_rho, uv_rotate_values.expected_rotated_v)


VerticalValues = namedtuple(
    'VerticalValues',
    ['u',
     'v',
     's_rho',
     'mask_rho',
     'mask_u',
     'mask_v',
     'zeta',
     'h',
     'hc',
     'cs_r',
     'vtransform_1',
     'vtransform_2',
     'num_eta',
     'num_xi',
     'num_sigma',
     'time_index',
     'target_depth_default',
     'target_depth_surface',
     'target_depth_deep',
     'expected_u_target_depth_vtransform1',
     'expected_v_target_depth_vtransform1',
     'expected_u_target_depth_vtransform2',
     'expected_v_target_depth_vtransform2',
     'expected_u_target_depth_surface',
     'expected_v_target_depth_surface',
     'expected_u_target_depth_deep',
     'expected_v_target_depth_deep'])


@pytest.fixture
def vertical_values():

    time_index = 0

    s_rho = numpy.array(

            [-0.975, -0.925, -0.875, -0.825, -0.775, -0.725, -0.675, -0.625, -0.575, -0.525, -0.475, -0.425, -0.375,
             -0.325, -0.275, -0.225, -0.175, -0.125, -0.075, -0.025]
        )

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
             [-0.12864403426647186, -0.14699482917785645, -0.1533738672733307]],

            [[-0.22195807099342346, -0.2421712726354599, -0.2541458308696747],
             [-0.2059989869594574, -0.20967665314674377, -0.22562970221042633],
             [-0.13121622800827026, -0.14990080893039703, -0.15639856457710266]],

            [[-0.22682039439678192, -0.24759206175804138, -0.2597068250179291],
             [-0.21037042140960693, -0.21409402787685394, -0.23025770485401154],
             [-0.1335427612066269, -0.15248820185661316, -0.15910528600215912]],

            [[-0.23106896877288818, -0.2523146867752075, -0.2645571827888489],
             [-0.21420377492904663, -0.21799913048744202, -0.23432427644729614],
             [-0.13567526638507843, -0.15483982861042023, -0.16156552731990814]],

            [[-0.2349419891834259, -0.25623324513435364, -0.26865893602371216],
             [-0.21740630269050598, -0.22135674953460693, -0.23781266808509827],
             [-0.137639582157135, -0.15701685845851898, -0.16382475197315216]],

            [[-0.23808951675891876, -0.25947651267051697, -0.2720396816730499],
             [-0.2200968861579895, -0.2241903692483902, -0.24075093865394592],
             [-0.13947321474552155, -0.15906816720962524, -0.16592034697532654]],

            [[-0.24045346677303314, -0.2621067762374878, -0.2748018801212311],
             [-0.22281144559383392, -0.22653259336948395, -0.24322162568569183],
             [-0.14117150008678436, -0.1610313355922699, -0.16789506375789642]],

            [[-0.2426738142967224, -0.26430219411849976, -0.2770835757255554],
             [-0.22536149621009827, -0.22852501273155212, -0.24533599615097046],
             [-0.14282099902629852, -0.16296876966953278, -0.16980309784412384]],

            [[-0.24465344846248627, -0.26616188883781433, -0.27903714776039124],
             [-0.22754064202308655, -0.2302609086036682, -0.2472214251756668],
             [-0.14445151388645172, -0.16493166983127594, -0.17171379923820496]],

            [[-0.2462834268808365, -0.2679187059402466, -0.2808115780353546],
             [-0.2292623072862625, -0.23193377256393433, -0.24901340901851654],
             [-0.14620362222194672, -0.16701747477054596, -0.17371375858783722]],

            [[-0.24794217944145203, -0.26969125866889954, -0.2825746238231659],
             [-0.23105888068675995, -0.23365876078605652, -0.2508770823478699],
             [-0.14814794063568115, -0.1692989468574524, -0.17590203881263733]],

            [[-0.24987787008285522, -0.271731436252594, -0.28456488251686096],
             [-0.23317913711071014, -0.23567895591259003, -0.2530546188354492],
             [-0.15038220584392548, -0.17185862362384796, -0.17835474014282227]],

            [[-0.25258249044418335, -0.27453669905662537, -0.28726720809936523],
             [-0.23601099848747253, -0.23847302794456482, -0.2559637129306793],
             [-0.15316414833068848, -0.17494668066501617, -0.18129220604896545]],

            [[-0.25681057572364807, -0.2789645791053772, -0.29178279638290405],
             [-0.2401619553565979, -0.24281151592731476, -0.26032811403274536],
             [-0.157413512468338, -0.17946366965770721, -0.18561050295829773]]

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
             [-0.004172911401838064, -0.0034219594672322273, 0.009713913314044476]],

            [[-0.01949373073875904, -0.01808745414018631, -0.01643970049917698],
             [-0.017525114119052887, -0.005732164718210697, -0.0002435247297398746],
             [-0.004358743317425251, -0.0035946646239608526, 0.009800488129258156]],

            [[-0.019242193549871445, -0.01788627728819847, -0.015856854617595673],
             [-0.01744634471833706, -0.005447198636829853, 0.00035470310831442475],
             [-0.0045697917230427265, -0.0038014191668480635, 0.009822248481214046]],

            [[-0.01894020289182663, -0.017642680555582047, -0.015223790891468525],
             [-0.0173249002546072, -0.005153473932296038, 0.0009491514065302908],
             [-0.004811229649931192, -0.004046828951686621, 0.009775890968739986]],

            [[-0.018637115135788918, -0.017389029264450073, -0.014595868065953255],
             [-0.017193809151649475, -0.004880314692854881, 0.0014993855729699135],
             [-0.005084861535578966, -0.0043313330970704556, 0.00966064352542162]],

            [[-0.018358834087848663, -0.017158253118395805, -0.014026599936187267],
             [-0.01708066090941429, -0.004655276890844107, 0.0019702124409377575],
             [-0.00539231114089489, -0.004652077332139015, 0.009479362517595291]],

            [[-0.018139295279979706, -0.016974788159132004, -0.013552641496062279],
             [-0.017011547461152077, -0.004495992790907621, 0.002341251587495208],
             [-0.005724799819290638, -0.005007237195968628, 0.009235722944140434]],

            [[-0.01799042522907257, -0.016854379326105118, -0.013189395889639854],
             [-0.01699862815439701, -0.004410982131958008, 0.0026074869092553854],
             [-0.006096957251429558, -0.0053996010683476925, 0.00892884936183691]],

            [[-0.01792767457664013, -0.016808394342660904, -0.01293881144374609],
             [-0.017050525173544884, -0.004404849372804165, 0.002771394792944193],
             [-0.006511836312711239, -0.005842543672770262, 0.008547094650566578]],

            [[-0.017963677644729614, -0.016854533925652504, -0.012803212739527225],
             [-0.01718880981206894, -0.004488971084356308, 0.002831200836226344],
             [-0.007028862368315458, -0.0063656410202383995, 0.00805988721549511]],

            [[-0.01813627779483795, -0.017030464485287666, -0.012803859077394009],
             [-0.017449606209993362, -0.0046984232030808926, 0.0027638953179121017],
             [-0.007689429447054863, -0.007028832100331783, 0.007404951844364405]],

            [[-0.018535180017352104, -0.01742624118924141, -0.013013389892876148],
             [-0.017937809228897095, -0.0051326327957212925, 0.0024881192948669195],
             [-0.008611081168055534, -0.007948571816086769, 0.006469985470175743]],

            [[-0.019416848197579384, -0.01828826777637005, -0.013652382418513298],
             [-0.018961360678076744, -0.006087823770940304, 0.0017477537039667368],
             [-0.010075435042381287, -0.009396139532327652, 0.005024670157581568]],

            [[-0.021592704579234123, -0.02032707817852497, -0.01557872723788023],
             [-0.021354755386710167, -0.008356819860637188, -0.0003598624316509813],
             [-0.012770703993737698, -0.012113306671380997, 0.0024051270447671413]]
        ]])

    mask_rho = numpy.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]

        ])

    mask_u = numpy.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]

        ])

    mask_v = numpy.array(
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

    h = numpy.array(
        [
            [2, 3.527, 4.8061],
            [4.2, 4.5, 0.2],
            [5.5952, 12.5469, 7.042]

        ])

    hc = 2

    cs_r = numpy.array(

            [-0.992022610389818, -0.975604774961717, -0.957016056280816, -0.933938399902067, -0.903417908722867,
             -0.861896431852144, -0.805639548819673, -0.731827045640921, -0.640306443956229, -0.535254642140977,
             -0.425235513534097, -0.320717020060839, -0.230290147888241, -0.158187399435204, -0.104343218451353,
             -0.0660600501546584, -0.0397681062338454, -0.0221242106368482, -0.0104507128719878, -0.00278112096054041]
        )

    vtransform_1 = 1
    vtransform_2 = 2

    num_eta = 3
    num_xi = 3
    num_sigma = 20

    target_depth_default = 4.5
    target_depth_surface = 0
    target_depth_deep = 10

    expected_u_target_depth_vtransform1 = numpy.array(
        [

            [-0.20652716, -0.24294458, -0.2589181],
            [-0.20850541, -0.2128753, -0.24432442],
            [-0.1376805, -0.15896076, -0.1634732]

        ])

    expected_v_target_depth_vtransform1 = numpy.array(
        [

            [-0.01971084, -0.01805876, -0.01593952],
            [-0.01747995, -0.00552582, 0.00212063],
            [-0.00509172, -0.00463528, 0.00967858]

        ])

    expected_u_target_depth_vtransform2 = numpy.array(
        [

            [-0.21491652, -0.24441181, -0.25952847],
            [-0.20921896, -0.21346418,  0.03362792],
            [-0.13758148, -0.15891376, -0.1634139]

        ])

    expected_v_target_depth_vtransform2 = numpy.array(
        [

            [-0.01968389, -0.0180043, -0.01587555],
            [-0.01746709, -0.00548783, -0.00343518],
            [-0.00507677, -0.00462793,  0.0096816]

        ])

    expected_u_target_depth_surface = numpy.array(
        [

            [-0.24533438, -0.26770889, -0.28122804],
            [-0.22935442, -0.23218014, -0.1792364],
            [-0.17038436, -0.18940582, -0.19772761]

        ])

    expected_v_target_depth_surface = numpy.array(
        [

            [-0.01794272, -0.01684902, -0.01280337],
            [-0.01720218, -0.00451889, -0.00159006],
            [-0.0209978, -0.01809395, -0.00494529]

        ])

    expected_u_target_depth_deep = numpy.array(
        [

            [-0.20652716, -0.24294458, -0.2589181],
            [-0.20850541, -0.2128753, -0.24432442],
            [-0.1376805, -0.15611915, -0.1634732]

        ])

    expected_v_target_depth_deep = numpy.array(
        [

            [-0.01971084, -0.01805876, -0.01593952],
            [-0.01747995, -0.00552582, 0.00212063],
            [-0.00509172, -0.00421402, 0.00967858]

        ])

    return VerticalValues(u, v, s_rho, mask_rho, mask_u, mask_v, zeta, h, hc, cs_r, vtransform_1, vtransform_2, num_eta,
                          num_xi, num_sigma, time_index, target_depth_default, target_depth_surface, target_depth_deep,
                          expected_u_target_depth_vtransform1, expected_v_target_depth_vtransform1,
                          expected_u_target_depth_vtransform2, expected_v_target_depth_vtransform2,
                          expected_u_target_depth_surface, expected_v_target_depth_surface,
                          expected_u_target_depth_deep, expected_v_target_depth_deep)


def test_vertical_interpolation_vtransform1(vertical_values):
    """Test vertical interpolation of u/v to default target depth with
       vertical transformation equation 1.
    """
    u_target_depth, v_target_depth = vertical_interpolation(vertical_values.u, vertical_values.v, vertical_values.s_rho,
                                                            vertical_values.mask_rho, vertical_values.mask_u,
                                                            vertical_values.mask_v, vertical_values.zeta,
                                                            vertical_values.h, vertical_values.hc, vertical_values.cs_r,
                                                            vertical_values.vtransform_1, vertical_values.num_eta,
                                                            vertical_values.num_xi, vertical_values.num_sigma,
                                                            vertical_values.time_index,
                                                            vertical_values.target_depth_default)

    # print(f"u_target_depth_vtransform1: {u_target_depth}")
    # print(f"v_target_depth_vtransform1: {v_target_depth}")
    assert numpy.allclose(u_target_depth, vertical_values.expected_u_target_depth_vtransform1)
    assert numpy.allclose(v_target_depth, vertical_values.expected_v_target_depth_vtransform1)


def test_vertical_interpolation_vtransform2(vertical_values):
    """Test vertical interpolation of u/v to default target depth with
       vertical transformation equation 2.
    """
    u_target_depth, v_target_depth = vertical_interpolation(vertical_values.u, vertical_values.v, vertical_values.s_rho,
                                                            vertical_values.mask_rho, vertical_values.mask_u,
                                                            vertical_values.mask_v, vertical_values.zeta,
                                                            vertical_values.h, vertical_values.hc, vertical_values.cs_r,
                                                            vertical_values.vtransform_2, vertical_values.num_eta,
                                                            vertical_values.num_xi, vertical_values.num_sigma,
                                                            vertical_values.time_index,
                                                            vertical_values.target_depth_default)

    # print(f"u_target_depth_vtransform2: {u_target_depth}")
    # print(f"v_target_depth_vtransform2: {v_target_depth}")
    assert numpy.allclose(u_target_depth, vertical_values.expected_u_target_depth_vtransform2)
    assert numpy.allclose(v_target_depth, vertical_values.expected_v_target_depth_vtransform2)


def test_vertical_interpolation_at_surface(vertical_values):
    """Test vertical interpolation of u/v to surface target depth"""
    u_target_depth, v_target_depth = vertical_interpolation(vertical_values.u, vertical_values.v, vertical_values.s_rho,
                                                            vertical_values.mask_rho, vertical_values.mask_u,
                                                            vertical_values.mask_v, vertical_values.zeta,
                                                            vertical_values.h, vertical_values.hc, vertical_values.cs_r,
                                                            vertical_values.vtransform_1, vertical_values.num_eta,
                                                            vertical_values.num_xi, vertical_values.num_sigma,
                                                            vertical_values.time_index,
                                                            vertical_values.target_depth_surface)

    # print(f"u_target_depth_surface: {u_target_depth}")
    # print(f"v_target_depth_surface: {v_target_depth}")
    assert numpy.allclose(u_target_depth, vertical_values.expected_u_target_depth_surface)
    assert numpy.allclose(v_target_depth, vertical_values.expected_v_target_depth_surface)


def test_vertical_interpolation_deep(vertical_values):
    """Test vertical interpolation of u/v to deep target depth"""
    u_target_depth, v_target_depth = vertical_interpolation(vertical_values.u, vertical_values.v, vertical_values.s_rho,
                                                            vertical_values.mask_rho, vertical_values.mask_u,
                                                            vertical_values.mask_v, vertical_values.zeta,
                                                            vertical_values.h, vertical_values.hc, vertical_values.cs_r,
                                                            vertical_values.vtransform_1, vertical_values.num_eta,
                                                            vertical_values.num_xi, vertical_values.num_sigma,
                                                            vertical_values.time_index, vertical_values.target_depth_deep)

    # print(f"******u_target_depth_deep: {u_target_depth}")
    # print(f"******v_target_depth_deep: {v_target_depth}")
    assert numpy.allclose(u_target_depth, vertical_values.expected_u_target_depth_deep)
    assert numpy.allclose(v_target_depth, vertical_values.expected_v_target_depth_deep)
