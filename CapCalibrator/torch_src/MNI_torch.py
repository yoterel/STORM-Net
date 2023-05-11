import numpy as np
from pathlib import Path
import torch
import logging


def torch_find_affine_transforms(our_anchors_xyz, our_sensors_xyz, selected_indices, refN, pointN, resource_folder="resource"):
    """
    finds refN affine transforms between our anchors and anchors from all reference template brains
    :param our_anchors_xyz: our anchors
    :param our_sensors_xyz: our sensors (on head surface)
    :param selected_indices: our selected anchors out of the 23 10-20 points
    :param refN: number of reference brains in template data
    :return: numpy array of size refN x number_of_sensors x 3
    represents for each refernce brain all our sensors locations in its frame of reference
    """
    # path_wo_ext = "resource/MNI_templates/DMNIHAve"
    # dmnihavePath = Path(path_wo_ext + ".csv")
    # if Path(path_wo_ext+".npy").is_file():
    #     DMNIHAve = np.load(path_wo_ext+".npy", allow_pickle=True)
    # else:
    #     DMNIHAve = np.genfromtxt(dmnihavePath, delimiter=',')
    #     np.save(path_wo_ext, DMNIHAve)
    # template_anchors_xyz = DMNIHAve[selected_indices, :]
    # ==================== AffineEstimation4 ====================== (l 225)
    size = len(selected_indices)
    assert size >= 4
    device = our_anchors_xyz.device
    A = torch.cat((our_anchors_xyz, torch.ones((size, 1), device=device)), dim=-1)
    A = A.repeat(refN, 1).reshape((refN, size, 4))
    # load 17 brain templates from disk
    DMS = []
    for i in range(1, refN+1):
        path_wo_ext = resource_folder+"/MNI_templates/DMNI{:0>4d}".format(i)
        if Path(path_wo_ext+".npy").is_file():
            DMS.append(np.load(path_wo_ext+".npy", allow_pickle=True))
        else:
            csv_path = Path(path_wo_ext+".csv")
            DM = np.genfromtxt(csv_path, delimiter=',')
            np.save(path_wo_ext, DM)
            DMS.append(DM)
    B = torch.FloatTensor(DMS).to(device)
    B = B[:, selected_indices, :]
    B = torch.cat((B, torch.ones((refN, size, 1), device=device)), dim=-1)
    # W = A.pinverse() @ B
    W = torch.inverse(A.transpose(1, 2) @ A) @ (A.transpose(1, 2) @ B)
    # test = np.all(np.isclose(W, W_test, atol=1e-4))
    # W = torch.linalg.lstsq(A, B).solution
    # find affine transformation between our anchors and all brains
    DDDD = torch.cat((our_sensors_xyz, torch.ones((pointN, 1), device=device)), dim=-1)
    DDDD = DDDD.repeat(refN, 1).reshape((refN, pointN, 4))
    return torch.bmm(DDDD, W)[:, :, :3]


def k_softmin(k, x):
    nominator = torch.exp(-k * x)
    demonimator = torch.sum(torch.exp(-k * x), dim=-1, keepdim=True)
    return nominator / demonimator


def torch_find_closest_on_surface(others, refN, pointN, soft_dist_func="softkmin", resource_folder="resource"):
    k = 10
    new_others = torch.empty(others.shape, device=others.device)
    for i in range(refN):
        my_str = resource_folder+"/MNI_templates/xyzall{}.npy".format(str(i + 1))
        xyz = load_raw_MNI_data(my_str, i, resource_folder)
        xyz = torch.FloatTensor(xyz).to(others.device)
        single_instance = others[i].unsqueeze(1)
        distances = torch.linalg.norm(single_instance - xyz.repeat(85, 1, 1), dim=-1).double()
        # hard_min = torch.min(distances, dim=-1).values
        # other_soft_min = torch.topk(distances, 1000, dim=-1, largest=False)
        if soft_dist_func == "softkmin":
            x = k_softmin(k, distances)
            p = (x.float() @ xyz)
        else:
            raise NotImplementedError
        # p_real = xyz[torch.argmin(distances, dim=-1)]
        # print("error: ", torch.mean(torch.norm(p - p_real, dim=1)))
        new_others[i] = p
    return torch.mean(new_others, axis=0)


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
        xallbemPath = Path(resource_folder+"/MNI_templates/xall"+shortcut+".csv")
        yallbemPath = Path(resource_folder+"/MNI_templates/yall"+shortcut+".csv")
        zallbemPath = Path(resource_folder+"/MNI_templates/zall"+shortcut+".csv")
        xallBEM = np.genfromtxt(xallbemPath, delimiter=',')
        yallBEM = np.genfromtxt(yallbemPath, delimiter=',')
        zallBEM = np.genfromtxt(zallbemPath, delimiter=',')
        XYZ = np.column_stack((xallBEM, yallBEM, zallBEM))
        np.save(location, XYZ)
    return XYZ


def torch_project(origin_xyz, others_xyz, selected_indices, resource_folder="resource"):
    refN = 17  # number of reference brains
    batch_size = others_xyz.shape[0]
    pointN = others_xyz.shape[1]  # number of sensors to project
    # get sensors transformed into reference brains coordinate systems
    for i in range(batch_size):
        others_transformed_to_ref = torch_find_affine_transforms(origin_xyz,
                                                                 others_xyz[i],
                                                                 selected_indices,
                                                                 refN,
                                                                 pointN,
                                                                 resource_folder)
        if torch.any(torch.isnan(others_transformed_to_ref)):
            logging.info("nans in torch affine !!")
        # XYZ = load_raw_MNI_data("resource/MNI_templates/xyzallBEM.npy", "brain")
        # XYZ = torch.FloatTensor(XYZ, device=others_xyz.device)
        projected_sensors = torch_find_closest_on_surface(others_transformed_to_ref, refN, pointN, soft_dist_func="softkmin", resource_folder=resource_folder)
        if torch.any(torch.isnan(projected_sensors)):
            logging.info("nans in torch project !!")
        others_xyz[i] = projected_sensors
    return others_xyz


def torch_find_affine_transforms_non_diff(our_anchors_xyz, our_sensors_xyz, selected_indices, refN, pointN, resource_folder="resource"):
    """
    finds refN affine transforms between our anchors and anchors from all reference template brains
    :param our_anchors_xyz: our anchors
    :param our_sensors_xyz: our sensors (on head surface)
    :param selected_indices: our selected anchors out of the 23 10-20 points
    :param refN: number of reference brains in template data
    :return: numpy array of size refN x number_of_sensors x 3
    represents for each refernce brain all our sensors locations in its frame of reference
    """
    # path_wo_ext = "resource/MNI_templates/DMNIHAve"
    # dmnihavePath = Path(path_wo_ext + ".csv")
    # if Path(path_wo_ext+".npy").is_file():
    #     DMNIHAve = np.load(path_wo_ext+".npy", allow_pickle=True)
    # else:
    #     DMNIHAve = np.genfromtxt(dmnihavePath, delimiter=',')
    #     np.save(path_wo_ext, DMNIHAve)
    # template_anchors_xyz = DMNIHAve[selected_indices, :]
    # ==================== AffineEstimation4 ====================== (l 225)
    size = len(selected_indices)
    assert size >= 4
    device = our_anchors_xyz.device
    A = torch.cat((our_anchors_xyz, torch.ones((size, 1), device=device)), dim=-1)
    A = A.repeat(refN, 1).reshape((refN, size, 4))
    # load 17 brain templates from disk
    DMS = []
    for i in range(1, refN+1):
        path_wo_ext = resource_folder+"/MNI_templates/DMNI{:0>4d}".format(i)
        if Path(path_wo_ext+".npy").is_file():
            DMS.append(np.load(path_wo_ext+".npy", allow_pickle=True))
        else:
            csv_path = Path(path_wo_ext+".csv")
            DM = np.genfromtxt(csv_path, delimiter=',')
            np.save(path_wo_ext, DM)
            DMS.append(DM)
    B = torch.FloatTensor(DMS).to(device)
    B = B[:, selected_indices, :]
    B = torch.cat((B, torch.ones((refN, size, 1), device=device)), dim=-1)
    W = torch.linalg.lstsq(A, B).solution
    # W = A.pinverse() @ B
    # W = torch.linalg.lstsq(A, B).solution
    # find affine transformation between our anchors and all brains
    DDDD = torch.cat((our_sensors_xyz, torch.ones((pointN, 1), device=device)), dim=-1)
    DDDD = DDDD.repeat(refN, 1).reshape((refN, pointN, 4))
    return torch.bmm(DDDD, W)[:, :, :3]


def torch_find_closest_on_surface_naive(othersRefList, XYZ, pointN, calc_sd_and_var=False):
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
    device = othersRefList.device
    other = torch.ones((pointN, 3)).to(device)
    otherVar = torch.ones((pointN, 4)).to(device)
    otherSD = torch.ones((pointN, 4)).to(device)
    top = 3
    for i in range(pointN):
        AA = torch.mean(othersRefList[:, i], dim=0)
        PP = torch.broadcast_to(AA, XYZ.shape)
        D = torch.linalg.norm(XYZ - PP, dim=1)
        XYZtop = XYZ[torch.topk(D, largest=False, k=top).indices, :]
        closest = torch.mean(XYZtop, dim=0)
        other[i, :] = closest
        if calc_sd_and_var:
            AAA = othersRefList[:, i]
            AV = closest
            N = AAA.shape[0]
            subMat = torch.ones(AAA.shape).to(device)
            for j in range(N):
                subMat[j, :] = AV

            dispEach = AAA - subMat
            dispEachSq = dispEach * dispEach
            XYZSS = torch.sum(dispEachSq, dim=0)
            RSS = torch.sum(XYZSS)

            XYZVar = XYZSS / (N - 1)
            RSSVar = RSS / (N - 1)

            VV = torch.cat((XYZVar, RSSVar.unsqueeze(0)))
            otherVar[i, :] = VV
            otherSD[i, :] = torch.sqrt(VV)
    return other, otherVar, otherSD


def torch_find_closest_on_surface_full(othersRefList, refN, pointN, resource_folder):
    """
    full implementation of cortical projection using the balloon inflation algorithm described in
    https://doi.org/10.1016/j.neuroimage.2005.01.018
    :param othersRefList: the refN x pointN x 3 transformed sensor locations (into ref brains)
    :param refN: number of reference brains
    :param pointN: number of sensors
    :return:
    """
    device = othersRefList.device
    otherRefCList = torch.empty((refN, pointN, 3), dtype=torch.float).to(device)
    # otherRefCList = np.empty((1, refN), dtype=object)
    for i in range(refN):
        my_str = resource_folder+"/MNI_templates/xyzall{}.npy".format(str(i + 1))
        XYZ = load_raw_MNI_data(my_str, i, resource_folder=resource_folder)
        XYZ = torch.FloatTensor(XYZ).to(othersRefList.device).to(device)
        projectionListC = torch.ones((pointN, 3)).to(device)
        for j in range(pointN):
            P = othersRefList[i, j, :3]
            PP = torch.broadcast_to(P, XYZ.shape)
            D = torch.linalg.norm(XYZ - PP, dim=1)
            top = round(XYZ.shape[0] * 0.05)  # select 5% of data (original paper selects 1000 points)
            XYZtop = XYZ[torch.topk(D, largest=False, k=top).indices, :]
            Nclose = 200
            XYZclose = XYZ[torch.topk(D, largest=False, k=Nclose).indices, :]
            PNear = torch.mean(XYZclose, dim=0)  # select mean of closest 200 points

            # Line between P and PNear
            p1 = P.unsqueeze(0)
            p2 = PNear.unsqueeze(0)
            p3 = XYZtop
            # cross product the line with the point and normalize gives exactly distance from line
            distance_from_line = torch.linalg.norm(torch.cross(torch.broadcast_to(p2 - p1, p3.shape), p3 - p1) / torch.linalg.norm(p2-p1), dim=1)
            det = 0
            rodR = 0
            while det == 0:
                rodR += 1
                Iless2 = torch.where(distance_from_line <= rodR)
                rod = XYZtop[Iless2]
                det = torch.sum(rod ** 2)

            # Find brain surface points on the vicinity of P (l 862)
            PPB = torch.broadcast_to(P, rod.shape)
            VicD = torch.linalg.norm(rod - PPB, dim=1)
            NVic = 3
            if rod.shape[0] < NVic:
                NVic = rod.shape[0]
            IVicD = torch.argsort(VicD)
            NIVicD = IVicD[0:NVic]
            Vic = rod[NIVicD, :]
            CP = torch.mean(Vic, dim=0)

            projectionListC[j, :] = CP
        otherRefCList[i] = projectionListC
    return otherRefCList


def torch_project_non_differentiable(origin_xyz, others_xyz, selected_indices, output_errors=False, resource_folder="resource"):
    """
    projects others_xyz to MNI coordiantes given anchors in origin_xyz
    :param origin_xyz: anchors given as nx3 np array (n >= 4)
    :param others_xyz: optodes to project given as bxmx3 np array (b is batch size, m is number of optodes)
    :param selected_indices: which indices to select from origin_xyz as anchors given as np array (len must be at least 4)
                             order matters! selection is based on this order:
                             ["nz", "iz", "rpa", "lpa",
                              "fp1", "fp2", "fz", "f3",
                              "f4", "f7", "f8", "cz",
                              "c3", "c4", "t3", "t4",
                              "pz", "p3", "p4", "t5",
                              "t6", "o1", "o2"]
    :param output_errors: whether to output error in estimation as well.
    :param resource_folder: relative path to the fodler with the raw template data
    :return: others_batched - others transformed to MNI of ideal head  (cortical surface)
             others_sd_batched - standard deviation error per axis, point manner (for others_batched).
                        Last channel is SD across all axes (root sum of squares).
             ref - others projected onto cortical surface without nearest neighbor search on the average template
    """
    refN = 17  # number of reference brains
    batch_size = others_xyz.shape[0]  # number of sensors to project
    pointN = others_xyz.shape[1]  # number of sensors to project
    others_batched = torch.empty((batch_size, pointN, 3))
    others_sd_batched = torch.empty((batch_size, pointN, 4))
    for i in range(batch_size):
        logging.info("projected {} / {} batches".format(i, batch_size))
        others_transformed_to_ref = torch_find_affine_transforms_non_diff(origin_xyz,
                                                           others_xyz[i],
                                                           selected_indices,
                                                           refN,
                                                           pointN,
                                                           resource_folder)
        others_projected_to_ref = torch_find_closest_on_surface_full(others_transformed_to_ref, refN, pointN, resource_folder=resource_folder)
        XYZ = load_raw_MNI_data(resource_folder+"/MNI_templates/xyzallBEM.npy", "brain", resource_folder=resource_folder)
        XYZ = torch.FloatTensor(XYZ).to(others_xyz.device)
        # get closest points of projected sensors on average cortical surface
        otherC, otherCV, otherCSD = torch_find_closest_on_surface_naive(others_projected_to_ref, XYZ, pointN, output_errors)
        others_batched[i] = otherC
        others_sd_batched[i] = otherCSD
    return others_batched.squeeze(), others_sd_batched.squeeze(), torch.mean(others_projected_to_ref, dim=0)
