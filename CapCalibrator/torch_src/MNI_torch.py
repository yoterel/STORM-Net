import numpy as np
from pathlib import Path
import torch


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
    B = torch.FloatTensor(DMS, device=device)
    B = B[:, selected_indices, :]
    B = torch.cat((B, torch.ones((refN, size, 1), device=device)), dim=-1)
    W = A.pinverse() @ B
    # W = torch.linalg.lstsq(A, B).solution
    # find affine transformation between our anchors and all brains
    DDDD = torch.cat((our_sensors_xyz, torch.ones((pointN, 1), device=device)), dim=-1)
    DDDD = DDDD.repeat(refN, 1).reshape((refN, pointN, 4))
    return torch.bmm(DDDD, W)[:, :, :3]


def k_softmin(k, x):
    nominator = torch.exp(-k * x)
    demonimator = torch.sum(torch.exp(-k * x), dim=-1, keepdim=True)
    return nominator / demonimator


def find_closest_on_surface_differentiable(others, refN, pointN, soft_mask_func="softkmin", resource_folder="resource"):
    k = 10
    eps = 1e-5
    new_others = torch.empty(others.shape, device=others.device)
    for i in range(refN):
        my_str = resource_folder+"/MNI_templates/xyzall{}.npy".format(str(i + 1))
        xyz = load_raw_MNI_data(my_str, i, resource_folder)
        xyz = torch.FloatTensor(xyz, device=others.device)
        single_instance = others[i].unsqueeze(1)
        distances = torch.linalg.norm(single_instance - xyz.repeat(85, 1, 1), dim=-1).double()
        # hard_min = torch.min(distances, dim=-1).values
        soft_min = -torch.log(torch.sum(torch.exp(-k*distances), dim=-1)) / k
        # other_soft_min = torch.topk(distances, 1000, dim=-1, largest=False)
        if soft_mask_func == "softkmin":
            x = (torch.abs(soft_min.unsqueeze(1) - distances))
            soft_mask = k_softmin(k, x)
            denom = 1
        elif soft_mask_func == "log":
            soft_mask = -torch.log((torch.abs(soft_min.unsqueeze(1) - distances) + eps))
            soft_mask = torch.clip(soft_mask, min=0)
            denom = torch.sum(soft_mask, dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        p = (soft_mask.float() @ xyz) / denom
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
            print("what the ?")
        # XYZ = load_raw_MNI_data("resource/MNI_templates/xyzallBEM.npy", "brain")
        # XYZ = torch.FloatTensor(XYZ, device=others_xyz.device)
        projected_sensors = find_closest_on_surface_differentiable(others_transformed_to_ref, refN, pointN, resource_folder=resource_folder)
        if torch.any(torch.isnan(projected_sensors)):
            print("what the ?")
        others_xyz[i] = projected_sensors
    return others_xyz
