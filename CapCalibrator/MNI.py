import numpy as np
from pathlib import Path


def find_affine_transforms(our_anchors_xyz, our_sensors_xyz, selected_indices, refN, pointN, resource_folder="resource"):
    """
    finds refN affine transforms between our anchors and anchors from all reference template brains
    :param our_anchors_xyz: our anchors
    :param our_sensors_xyz: our sensors (on head surface)
    :param selected_indices: our selected anchors out of the 23 10-20 points
    :param refN: number of reference brains in template data
    :return: numpy array of size refN x number_of_sensors x 3
    represents for each refernce brain all our sensors locations in its frame of reference
    """
    # todo: refactor this function to be more readable

    # ==================== AffineEstimation4 ======================
    size = len(selected_indices)
    assert size >= 4
    # find affine transformation with ideal brain (not used anywhere..)
    listOri = np.c_[our_anchors_xyz, np.ones(size)]
    # ------------ Transformation to reference brains --------------
    # find affine transformation with every brain in the 17 templates
    refBList = np.empty((refN, 2), dtype=object)
    # load 17 brain templates from disk
    DMS = []
    resource_folder = str(resource_folder)
    for i in range(1, refN+1):
        path_wo_ext = resource_folder + "/MNI_templates/DMNI{:0>4d}".format(i)
        if Path(path_wo_ext+".npy").is_file():
            DMS.append(np.load(path_wo_ext+".npy", allow_pickle=True))
        else:
            csv_path = Path(path_wo_ext+".csv")
            DM = np.genfromtxt(csv_path, delimiter=',')
            np.save(path_wo_ext, DM)
            DMS.append(DM)
    # find affine transformation between our anchors and all brains
    for i in range(1, refN+1):
        DM = DMS[i-1][selected_indices, :]
        refDist = np.c_[DM, np.ones(size)]
        WW = np.linalg.lstsq(listOri, refDist, rcond=None)[0]
        refBList[i-1, 0] = np.matmul(listOri, WW)
        refBList[i-1, 1] = WW
    # ---------- Transforming given head surface points stored in others to the ideal brain and each ref brain -----
    DDDD = np.c_[our_sensors_xyz, np.ones(pointN)]
    othersRefList = np.empty((refN, pointN, DDDD.shape[1]), dtype=np.float)
    originRegList = np.empty((refN, our_anchors_xyz.shape[0], DDDD.shape[1]), dtype=np.float)
    for i in range(refN):
        WR = refBList[i, 1]
        othersRef = np.matmul(DDDD, WR)
        othersRefList[i] = othersRef
        originRegList[i] = refBList[i, 0]
    affine_transforms = np.stack([x for x in refBList[:, 1]])
    return affine_transforms, othersRefList[:, :, :3], originRegList[:, :, :3]


def find_closest_on_surface_naive(othersRefList, XYZ, pointN, calc_sd_and_var=False):
    """
    finds closest point on cortical surface for every (transformed) sensor location
    by averaging over 3 closest points on cortical surface
    :param othersRefList: the refN x pointN x 3 transformed sensor locations (into ref brains)
    :param XYZ the raw measurements from template reference brains
    :param pointN: number of sensors
    :return:
    other - location on cortical surface per sensor
    otherVar - variance of each otherH sensor
    otherSD - root of variance of each otherH sensor
    """
    other = np.ones((pointN, 3))
    otherVar = np.ones((pointN, 4))
    otherSD = np.ones((pointN, 4))
    top = 3
    for i in range(pointN):
        AA = np.mean(othersRefList[:, i], axis=0)
        # ----- Back projection -----
        PP = np.broadcast_to(AA, XYZ.shape)
        D = np.linalg.norm(XYZ - PP, axis=1)
        IDtop = np.argpartition(D, top)[:top]  # sort by lowest norm
        XYZtop = XYZ[IDtop, :]
        closest = np.mean(XYZtop, axis=0)
        # -------- End of back projection ----------

        other[i, :] = closest

        # ---- Variance calculation ----
        if calc_sd_and_var:
            AAA = othersRefList[:, i]
            AV = closest
            N = AAA.shape[0]
            subMat = np.ones(AAA.shape)
            for j in range(N):
                subMat[j, :] = AV

            dispEach = AAA - subMat
            dispEachSq = dispEach * dispEach
            XYZSS = np.sum(dispEachSq, axis=0)
            RSS = np.sum(XYZSS)

            XYZVar = XYZSS / (N - 1)
            RSSVar = RSS / (N - 1)

            VV = np.append(XYZVar, RSSVar)
            otherVar[i, :] = VV
            otherSD[i, :] = np.sqrt(VV)
    return other, otherVar, otherSD


def find_closest_on_surface_full(othersRefList, refN, pointN, resource_folder):
    """
    full implementation of cortical projection using the balloon inflation algorithm described in
    https://doi.org/10.1016/j.neuroimage.2005.01.018
    :param othersRefList: the refN x pointN x 3 transformed sensor locations (into ref brains)
    :param refN: number of reference brains
    :param pointN: number of sensors
    :return:
    """
    otherRefCList = np.empty((refN, pointN, 3), dtype=np.float)
    for i in range(refN):
        my_str = resource_folder+"/MNI_templates/xyzall{}.npy".format(str(i + 1))
        XYZ = load_raw_MNI_data(my_str, i, resource_folder=resource_folder)
        projectionListC = np.ones((pointN, 3))
        for j in range(pointN):
            P = othersRefList[i, j, :3]
            PP = np.broadcast_to(P, XYZ.shape)
            D = np.linalg.norm(XYZ - PP, axis=1)
            top = round(XYZ.shape[0] * 0.05)  # select 5% of data (original paper selects 1000 points)
            IDtop = np.argpartition(D, top)[:top]  # sort by lowest norm
            XYZtop = XYZ[IDtop, :]
            Nclose = 200
            IDclose = np.argpartition(D, Nclose)[:Nclose]  # sort by lowest norm
            XYZclose = XYZ[IDclose, :]
            PNear = np.mean(XYZclose, axis=0)  # select mean of closest 200 points
            # Line between P and PNear
            p1 = P
            p2 = PNear
            p3 = XYZtop
            # cross product the line with the point and normalize gives exactly distance from line
            distance_from_line = np.linalg.norm(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2-p1), axis=1)
            # Extend the line P-Pnear to a rod (l 851)
            det = 0
            rodR = 0
            while det == 0:
                rodR += 1
                Iless2 = np.where(distance_from_line <= rodR)
                rod = XYZtop[Iless2, :][0]
                det = np.sum(rod ** 2)
            # Find brain surface points on the vicinity of P (l 862)
            PPB = np.broadcast_to(P, rod.shape)
            VicD = np.linalg.norm(rod - PPB, axis=1)
            NVic = 3
            if rod.shape[0] < NVic:
                NVic = rod.shape[0]
            IVicD = np.argsort(VicD)
            NIVicD = IVicD[0:NVic]
            Vic = rod[NIVicD, :]
            CP = np.mean(Vic, axis=0)
            projectionListC[j, :] = CP
        otherRefCList[i] = projectionListC
    return otherRefCList


def load_raw_MNI_data(location, type, resource_folder):
    """
    loads raw MNi data from disk
    :param location: where is the data located
    :param type: what does the data represent
     "brain" = average brain surface,
     "head" = average head surface,
      or number indicating reference brain index (brain surface data))
    :return:
    """
    if type == "brain":
        shortcut = "BEM"
    elif type == "head":
        shortcut = "HEM"
    else:
        shortcut = "M0" + str(type + 1)
    my_str = location
    if Path(my_str).is_file():
        XYZ = np.load(my_str, allow_pickle=True)
    else:
        resource_folder = str(resource_folder)
        xallbemPath = Path(resource_folder+"/MNI_templates/xall"+shortcut+".csv")
        yallbemPath = Path(resource_folder+"/MNI_templates/yall"+shortcut+".csv")
        zallbemPath = Path(resource_folder+"/MNI_templates/zall"+shortcut+".csv")
        xallBEM = np.genfromtxt(xallbemPath, delimiter=',')
        yallBEM = np.genfromtxt(yallbemPath, delimiter=',')
        zallBEM = np.genfromtxt(zallbemPath, delimiter=',')
        XYZ = np.column_stack((xallBEM, yallBEM, zallBEM))
        np.save(location, XYZ)
    return XYZ


def project(origin_xyz, others_xyz, selected_indices, output_errors=False, resource_folder="resource"):
    """
    projects others_xyz to MNI coordiantes given anchors in origin_xyz
    :param origin_xyz: anchors given as nx3 np array (n >= 4)
    :param others_xyz: optodes to project given as mx3 np array (m>=1)
    :param selected_indices: which indices to select from origin_xyz as anchors given as np array (len must be at least 4)
                             order matters! selection is based on this order:
                             ["nz", "iz", "rpa", "lpa",
                              "fp1", "fp2", "fz", "f3",
                              "f4", "f7", "f8", "cz",
                              "c3", "c4", "t3", "t4",
                              "pz", "p3", "p4", "t5",
                              "t6", "o1", "o2"]
    :param output_errors: whether to output error in estimation as well.
    :param resource_folder: relative path to the folder with the raw template data
    :return: otherH - others transformed to MNI of ideal head (head surface)
             otherC - others transformed to MNI of ideal head  (cortical surface)
             otherHSD - transformation standard deviation per axis, point manner (for otherH).
                        Last channel is SD across all axes (root sum of squares).
             otherCSD - transformation standard deviation per axis, point manner (for otherC).
                        Last channel is SD across all axes (root sum of squares).
    """
    resource_folder = str(resource_folder)
    refN = 17  # number of reference brains
    pointN = others_xyz.shape[0]  # number of sensors to project
    # get sensors transformed into reference brains coordinate systems
    transforms, others_transformed_to_ref, _ = find_affine_transforms(origin_xyz,
                                                                      others_xyz,
                                                                      selected_indices,
                                                                      refN,
                                                                      pointN,
                                                                      resource_folder)
    # load head surface raw data
    XYZ = load_raw_MNI_data(resource_folder+"/MNI_templates/xyzallHEM.npy", "head", resource_folder=resource_folder)
    # get closest location of sensors on average head surface
    otherH, otherHVar, otherHSD = find_closest_on_surface_naive(others_transformed_to_ref, XYZ, pointN, output_errors)
    # get location of sensors projected onto reference cortical surface by inflating a rod
    others_projected_to_ref = find_closest_on_surface_full(others_transformed_to_ref, refN, pointN, resource_folder=resource_folder)
    XYZ = load_raw_MNI_data(resource_folder+"/MNI_templates/xyzallBEM.npy", "brain", resource_folder=resource_folder)
    # get closest points of projected sensors on average cortical surface
    otherC, otherCVar, otherCSD = find_closest_on_surface_naive(others_projected_to_ref, XYZ, pointN, output_errors)
    # test, _, _ = find_closest_on_surface_naive(others_transformed_to_ref, XYZ, pointN)
    # SSwsH = otherHVar * (refN - 1)
    # SSwsC = otherCVar * (refN - 1)

    # ---- what he have by now ----
    # otherH - given head surface points transformed to the MNI ideal head (within-subject hat)
    # otherC - given cortical surface points transformed to the MNI ideal brain (within-subject hat)
    # otherHSD, otherCSD -  transformation SD for given head surface points, point manner
    # SSwsH, SSwsC - transformation SD for given cortical surface points, point manner
    # ----------------------
    return otherH, otherC, otherHSD, otherCSD, transforms


def vectorized_loop(XYZ, othersRefList, i, pointN):
    projectionListC = np.ones((pointN, 3))
    P = np.expand_dims(othersRefList[i, :, :3], axis=1)
    PP = np.broadcast_to(P, (pointN, XYZ.shape[0], 3))
    D = np.linalg.norm(XYZ - PP, axis=2)
    top = round(XYZ.shape[0] * 0.05)  # select 5% of data (original paper selects 1000 points)
    IDtop = np.argpartition(D, top, axis=1)[:, :top]  # sort by lowest norm
    XYZtop = XYZ[IDtop, :]

    Nclose = 200
    IDclose = np.argpartition(D, Nclose, axis=1)[:, :Nclose]  # sort by lowest norm
    XYZclose = XYZ[IDclose, :]
    PNear = np.mean(XYZclose, axis=1)  # select mean of closest 200 points
    # Line between P and PNear
    p1 = P
    p2 = np.expand_dims(PNear, axis=1)
    p3 = XYZtop
    # cross product the line with the point and normalize gives exactly distance from line
    distance_from_line = np.linalg.norm(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1, axis=2, keepdims=True), axis=2)
    condition = True
    rod_radius = 0
    while rod_radius < 100 and condition:
        rod_radius += 1
        test_rod_rad = np.where(distance_from_line <= rod_radius)
        diff = np.setdiff1d(np.arange(pointN), np.unique(test_rod_rad[0]))
        condition = len(diff) != 0
        # assert len(diff) == 0, "no point were found on cortex with a rod radius of <= 1 mm for sensor index {}".format(diff)
    for j in range(pointN):
        indices = np.where(test_rod_rad[0] == 0)[0]
        rod = XYZtop[j, test_rod_rad[1][indices]]
        # Find brain surface points on the vicinity of P (l 862)
        PPB = np.broadcast_to(P[j], rod.shape)
        VicD = np.linalg.norm(rod - PPB, axis=1)
        NVic = 3
        if rod.shape[0] < NVic:
            NVic = rod.shape[0]
        IVicD = np.argsort(VicD)
        NIVicD = IVicD[0:NVic]
        Vic = rod[NIVicD, :]
        CP = np.mean(Vic, axis=0)
        projectionListC[j, :] = CP
    return projectionListC
